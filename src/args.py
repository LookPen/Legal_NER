import logging
import os
from typing import Optional
from transformers import TrainingArguments


logger = logging.getLogger(__name__)


def initialize_traing_args(
        report_to: Optional[str] = "none",
        do_train: Optional[bool] = True,
        do_eval: Optional[bool] = True,
        do_predict: Optional[bool] = True,
        do_adv: Optional[bool] = False,
        seed: Optional[int] = 42,
        num_train_epochs: Optional[int] = 7,
        per_device_train_batch_size: Optional[int] = 8,
        per_device_eval_batch_size: Optional[int] = 8,
        learning_rate: Optional[float] = 5e-5,
        min_learning_rate: Optional[float] = None,
        output_dir: Optional[str] = "./output",
        evaluation_strategy: Optional[str] = "epoch",
        eval_steps: Optional[int] = 100,
        save_strategy: Optional[str] = "epoch",
        save_steps: Optional[int] = 100,
        remove_unused_columns: Optional[bool] = False,
        overwrite_output_dir: Optional[bool] = True,
        eval_accumulation_steps: Optional[int] = 5,
        fp16: Optional[bool] = False,
        dataloader_num_workers: Optional[int] = 8,
        save_total_limit: Optional[int] = 1,
        metric_for_best_model="f1"
):

    logger.info(f"The default output dir is {output_dir}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if "f1" in metric_for_best_model:
        metric_for_best_model = "f1"

    training_args = TrainingArguments(
        report_to=report_to,
        do_train=do_train,
        do_eval=do_eval,
        do_predict=do_predict,
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps if evaluation_strategy == "steps" else None,
        save_strategy=save_strategy,
        save_steps=eval_steps if save_strategy == "steps" else None,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        remove_unused_columns=remove_unused_columns,
        overwrite_output_dir=overwrite_output_dir,
        eval_accumulation_steps=eval_accumulation_steps,
        fp16=fp16,
        dataloader_num_workers=dataloader_num_workers
    )
    return training_args

