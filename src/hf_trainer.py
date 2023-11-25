
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A subclass of `Trainer` specific to  IE  tasks
"""
import os
import torch
import torch.nn as nn
import numpy as np
from transformers import Trainer, is_torch_tpu_available
from transformers.utils import (
    logging
)

from transformers.trainer_utils import (
    EvalLoopOutput,
    EvalPrediction,
    has_length,
    denumpify_detensorize,
)
from transformers.deepspeed import deepspeed_init
from transformers.trainer_pt_utils import (
    nested_concat,
    nested_numpify,
    nested_truncate,
    find_batch_size,
    IterableDatasetShard
)
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

from torch.utils.data import DataLoader


TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"

logger = logging.get_logger(__name__)



class HFTrainer(Trainer):
    # python中 *args 和 **kwargs的用法：https://blog.csdn.net/u010758410/article/details/71727822
    def __init__(self, *args, eval_examples=None, post_processor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_processor = post_processor

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (`List[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """
        train_output = super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)

        if self.args.save_total_limit == 1:
            import shutil
            for path in os.listdir(self.args.output_dir):
                if path.startswith("checkpoint"):
                    del_path = os.path.join(self.args.output_dir, path)
                    # print("Deleted old checkpoint dir...", del_path)
                    logger.info(f"Deleted old checkpoint dir [{del_path}]")
                    print(f"Deleted old checkpoint dir [{del_path}]")
                    shutil.rmtree(del_path)

        return train_output

    # def evaluation_loop(
    #     self,
    #     dataloader: DataLoader,
    #     description: str,
    #     prediction_loss_only: Optional[bool] = None,
    #     ignore_keys: Optional[List[str]] = None,
    #     metric_key_prefix: str = "eval",
    # ) -> EvalLoopOutput:
    #     """
    #     Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
    #
    #     Works both with or without labels.
    #     """
    #     args = self.args
    #
    #     prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only
    #
    #     # if eval is called w/o train init deepspeed here
    #     if args.deepspeed and not self.deepspeed:
    #
    #         # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
    #         # from the checkpoint eventually
    #         deepspeed_engine, _, _ = deepspeed_init(
    #             self, num_training_steps=0, resume_from_checkpoint=None, inference=True
    #         )
    #         self.model = deepspeed_engine.module
    #         self.model_wrapped = deepspeed_engine
    #         self.deepspeed = deepspeed_engine
    #
    #     model = self._wrap_model(self.model, training=False)
    #
    #     # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
    #     # while ``train`` is running, cast it to the right dtype first and then put on device
    #     if not self.is_in_train:
    #         if args.fp16_full_eval:
    #             model = model.to(dtype=torch.float16, device=args.device)
    #         elif args.bf16_full_eval:
    #             model = model.to(dtype=torch.bfloat16, device=args.device)
    #
    #     batch_size = self.args.per_device_eval_batch_size
    #
    #     logger.info(f"***** Running {description} *****")
    #     if has_length(dataloader):
    #         logger.info(f"  Num examples = {self.num_examples(dataloader)}")
    #     else:
    #         logger.info("  Num examples: Unknown")
    #     logger.info(f"  Batch size = {batch_size}")
    #
    #     model.eval()
    #
    #     self.callback_handler.eval_dataloader = dataloader
    #     # Do this before wrapping.
    #     eval_dataset = getattr(dataloader, "dataset", None)
    #
    #     if args.past_index >= 0:
    #         self._past = None
    #
    #     # Initialize containers
    #     # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
    #     losses_host = None
    #     preds_host = None
    #     labels_host = None
    #     # losses/preds/labels on CPU (final containers)
    #     all_losses = None
    #     all_preds = None
    #     all_labels = None
    #     # Will be useful when we have an iterable dataset so don't know its length.
    #
    #     observed_num_examples = 0
    #     # Main evaluation loop
    #     for step, inputs in enumerate(dataloader):
    #         # Update the observed num examples
    #         observed_batch_size = find_batch_size(inputs)
    #         if observed_batch_size is not None:
    #             observed_num_examples += observed_batch_size
    #             # For batch samplers, batch_size is not known by the dataloader in advance.
    #             if batch_size is None:
    #                 batch_size = observed_batch_size
    #
    #         # Prediction step
    #         loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
    #
    #         if is_torch_tpu_available():
    #             xm.mark_step()
    #
    #         # Update containers on host
    #         if loss is not None:
    #             losses = self._nested_gather(loss.repeat(batch_size))
    #             losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
    #         if labels is not None:
    #             labels = self._pad_across_processes(labels)
    #             labels = self._nested_gather(labels)
    #             labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
    #         if logits is not None:
    #             logits = self._pad_across_processes(logits)
    #             logits = self._nested_gather(logits)
    #             if self.preprocess_logits_for_metrics is not None:
    #                 logits = self.preprocess_logits_for_metrics(logits, labels)
    #             preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
    #         self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)
    #
    #         # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
    #         if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
    #             if losses_host is not None:
    #                 losses = nested_numpify(losses_host)
    #                 all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
    #             if preds_host is not None:
    #                 logits = nested_numpify(preds_host)
    #                 all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
    #             if labels_host is not None:
    #                 labels = nested_numpify(labels_host)
    #                 all_labels = (
    #                     labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
    #                 )
    #
    #             # Set back to None to begin a new accumulation
    #             losses_host, preds_host, labels_host = None, None, None
    #
    #     if args.past_index and hasattr(self, "_past"):
    #         # Clean the state at the end of the evaluation loop
    #         delattr(self, "_past")
    #
    #     # Gather all remaining tensors and put them back on the CPU
    #     if losses_host is not None:
    #         losses = nested_numpify(losses_host)
    #         all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
    #     if preds_host is not None:
    #         logits = nested_numpify(preds_host)
    #         all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
    #     if labels_host is not None:
    #         labels = nested_numpify(labels_host)
    #         all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
    #
    #     # Number of samples
    #     if has_length(eval_dataset):
    #         num_samples = len(eval_dataset)
    #     # The instance check is weird and does not actually check for the type, but whether the dataset has the right
    #     # methods. Therefore we need to make sure it also has the attribute.
    #     elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
    #         num_samples = eval_dataset.num_examples
    #     else:
    #         if has_length(dataloader):
    #             num_samples = self.num_examples(dataloader)
    #         else:  # both len(dataloader.dataset) and len(dataloader) fail
    #             num_samples = observed_num_examples
    #
    #     # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
    #     # samplers has been rounded to a multiple of batch_size, so we truncate.
    #     if all_losses is not None:
    #         all_losses = all_losses[:num_samples]
    #     if all_preds is not None:
    #         all_preds = nested_truncate(all_preds, num_samples)
    #     if all_labels is not None:
    #         all_labels = nested_truncate(all_labels, num_samples)
    #
    #     # Metrics!
    #     if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
    #         metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
    #     else:
    #         metrics = {}
    #
    #     # To be JSON-serializable, we need to remove numpy types or zero-d tensors
    #     metrics = denumpify_detensorize(metrics)
    #
    #     if all_losses is not None:
    #         metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
    #
    #     # Prefix all keys with metric_key_prefix + '_'
    #     for key in list(metrics.keys()):
    #         if not key.startswith(f"{metric_key_prefix}_"):
    #             metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
    #
    #     return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def evaluate(
            self,
            eval_dataset=None,
            eval_examples=None,
            ignore_keys=None,
            metric_key_prefix: str = "eval",
    ):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=False,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_processor is not None and self.compute_metrics is not None:
            eval_preds = self.post_processor.postprocess_predictions(
                eval_examples, eval_dataset, output.predictions,
                output_dir=self.args.output_dir,
                prefix="eval"
            )

            metrics = self.compute_metrics(eval_preds, eval_examples)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
            metrics.update(output.metrics)
            self.log(metrics)
        else:
            metrics = {}

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        return metrics

    def predict(
            self,
            predict_dataset,
            predict_examples,
            test_no_labels=True,
            ignore_keys=None,
            metric_key_prefix: str = "test",
    ):
        predict_dataloader = self.get_test_dataloader(predict_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )
        try:
            output = eval_loop(
                predict_dataloader,
                description="Prediction",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=False,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_processor is None:
            return output

        predictions = self.post_processor.postprocess_predictions(
            predict_examples, predict_dataset, output.predictions,
            output_dir=self.args.output_dir,
            prefix="test"
        )

        metric = {}
        if test_no_labels == False and self.compute_metrics is not None:
            metrics = self.compute_metrics(predictions, predict_examples)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
            metrics.update(output.metrics)
            self.log(metrics)

        return predictions, metric

    def predict_and_save(
            self,
            predict_dataset,
            predict_examples,
            over_stride=0,
            test_no_labels=True,
            ignore_keys=None,
            metric_key_prefix: str = "test",
    ):
        predictions, metric = self.predict(predict_dataset,
                                           predict_examples,
                                           test_no_labels=test_no_labels,
                                           ignore_keys=ignore_keys,
                                           metric_key_prefix=metric_key_prefix)
        # Save predictions
        import os
        training_args = self.args
        save_predictions_file = os.path.join(training_args.output_dir, f"final_predictions_{training_args.seed}.json")

        if self.is_world_process_zero():
            import json
            from collections import defaultdict

            results = defaultdict(list)
            for pred, example in zip(predictions, predict_examples):
                dataset_id, sent_id = example["example_ids"].split("_")
                results[dataset_id].append({"sent_id": sent_id,
                                            "entities": pred,
                                            "text": example["text"]})

            # 处理mrc模型的情况
            new_results = defaultdict(list)
            for dataset_id, v in results.items():
                unique_sent_ids = set([line["sent_id"] for line in v])
                tmp = {
                    sent_id: {
                        "sent_id": sent_id,
                        "text": "",
                        "entities": []
                    }
                    for sent_id in unique_sent_ids
                }
                for line in v:
                    tmp[line["sent_id"]]["text"] = line["text"]
                    tmp[line["sent_id"]]["entities"] += line["entities"]

                new_results[dataset_id] = list(tmp.values())

            final_result = []
            for preds in new_results.values():
                offset = 0
                entities = []
                text = ''
                preds = sorted(preds, key=lambda x: x["sent_id"])
                for i, line in enumerate(preds):
                    if i == 0:
                        text += line["text"]
                    else:
                        text += line["text"][over_stride:]

                    for ent in line["entities"]:
                        ent["start_idx"] = ent["start_idx"] + offset
                        ent["end_idx"] = ent["end_idx"] + offset
                        assert text[ent["start_idx"]:ent["end_idx"] + 1] == ent["entity"]
                        entities.append(ent)

                    offset += len(line["text"]) - over_stride
                final_result.append({
                    "text": text,
                    "entities": entities
                })

            json.dump(final_result, open(save_predictions_file, "w"), ensure_ascii=False, indent=4)

        return metric


    def load_checkpoint(self, model_path):
        state_dict = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state_dict)

