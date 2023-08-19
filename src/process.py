import collections
from typing import Optional, Tuple
import numpy as np
import os
import json
import logging

logger = logging.getLogger(__name__)


def fine_grade_tokenize(raw_text, tokenizer):
    """
        序列标注任务 BERT 分词器可能会导致标注偏移，
        用 char-level 来 tokenize,
        在使用之前，需要将bert的词表中的两个[unused]替换为[BLANK]，[INV]
    """
    tokens = []
    for _ch in raw_text:
        if _ch in [' ', '\t', '\n', '\r']:
            tokens.append(',')
        else:
            tokens.append(_ch)
    return tokens


class Processor:
    def __init__(self,
                 tokenizer,
                 ent_types,
                 querys=None,
                 max_seq_length=512
                 ):

        self.tokenizer = tokenizer
        self.ent_types = ent_types
        self.querys = querys
        if querys is None:
            self.ent2query = {ent:ent for ent in ent_types}
        else:
            self.ent2query = querys

        self.max_seq_length = max_seq_length
        self.over_stride = 30

    # def _get_ent2id(self):
    #     return {ent_type: idx for idx, ent_type in enumerate(self.ent_types)}

    def _get_ent2id(self):
        ent2id = {'O': 0}
        for ent_type in self.ent_types:
            ent2id['B+' + ent_type] = len(ent2id)
            ent2id['I+' + ent_type] = len(ent2id)
        return ent2id

    @classmethod
    def save_jsonl(cls, file_path, file, mode="w"):
        with open(file_path, mode, encoding='utf-8') as writer:
            writer.write(
                json.dumps(file, indent=4, ensure_ascii=False) + "\n"
            )

    # @property
    # def ent2query(self):
    #     if not hasattr(self, "_ent2query"):
    #         self._ent2query = {ent_type: query for ent_type, query in zip(self.ent_types, self.querys)}
    #     return self._ent2query

    @property
    def ent2id(self):
        if not hasattr(self, "_ent2id"):
            self._ent2id = self._get_ent2id()
        return self._ent2id

    @property
    def id2ent(self):
        return {v: k for k, v in self.ent2id.items()}

    def process_dataset(self, dataset, set_type="train", remove_columns=None, overwrite_cache=True):

        if "id" not in dataset.column_names:
            # 新增加一列id，用于后续将切分的长句子重新拼接成原句子。
            key_ = dataset.column_names[0]
            id_column = list(range(len(dataset[key_])))
            dataset = dataset.add_column("id", id_column)

        if remove_columns is None:
            remove_columns = dataset.column_names

        if "id" not in remove_columns:
            remove_columns.append("id")

        examples = dataset.map(
            self.get_examples,
            batched=True,
            batch_size=10,
            remove_columns=remove_columns,
            load_from_cache_file=not overwrite_cache,
            num_proc=1,
            desc=f"Running flatten & tagging on {set_type} dataset",
        )

        encoded_features = examples.map(
            self.convert_examples_to_features,
            batched=True,
            batch_size=10,
            load_from_cache_file=not overwrite_cache,
            remove_columns=examples.column_names,
            num_proc=1,
            desc=f"Running tokenizer on {set_type} dataset",
        )

        # encoded_features.set_format(type=self.features_type)

        return encoded_features, examples

    def get_examples(self, dataset):
        # 50相当于query length
        max_seq_length = self.max_seq_length - 50

        batch_texts = []
        batch_labels = []
        batch_exp_ids = []
        batch_ent_types = []
        batch_querys = []
        over_stride = self.over_stride
        for text_id, text, entities in zip(dataset["id"], dataset["text"], dataset["entities"]):

            if isinstance(entities, dict):
                entities = [{"start_idx": s, "end_idx": e, "type": t, "entity": ent} for s, e, t, ent in
                            zip(entities["start_idx"], entities["end_idx"], entities["type"], entities["entity"])]

            text = text.replace(" ", ",")
            # sub_sents = get_sub_seq_from_sentence(text, max_seq_length, over_stride=over_stride)
            sub_sents = [text]
            start_index = 0
            for sent_id, sent in enumerate(sub_sents):
                if len(sent) == 0:
                    continue
                # new_labels = refactor_labels(sent, entities, start_index)
                start_index += len(sent) - over_stride

                for ent_type, query in self.ent2query.items():
                    batch_querys.append(query)
                    batch_ent_types.append(ent_type)
                    batch_texts.append(sent)
                    batch_labels.append([ent for ent in entities if ent["type"] == ent_type])
                    batch_exp_ids.append("{}_{}".format(text_id, sent_id))

        examples = {
            "text": batch_texts,
            "labels": batch_labels,
            "example_ids": batch_exp_ids,
            "ent_types": batch_ent_types,
            "querys": batch_querys
        }
        return examples

    # tokenized examples
    def convert_examples_to_features(
            self,
            examples,
            text_column_name="text",
            labels_column_name="labels",
            ent_type_column_name="ent_types",
            query_column_name="querys"
    ):
        tokenizer = self.tokenizer
        ent2id = self.ent2id
        max_seq_length = self.max_seq_length

        all_tokenized_examples = collections.defaultdict(list)
        for text, labels, ent_type, query in zip(examples[text_column_name],
                                                 examples[labels_column_name],
                                                 examples[ent_type_column_name],
                                                 examples[query_column_name]):

            query_tokens = fine_grade_tokenize(query, tokenizer)
            content_tokens = fine_grade_tokenize(text, tokenizer)

            if len(query_tokens) + len(
                    content_tokens) > max_seq_length - 3:  # [cls] + query + [sep] + content + [sep]
                content_tokens = content_tokens[:max_seq_length - len(query_tokens) - 3]

            tokenized_examples = tokenizer(
                query_tokens,
                text_pair=content_tokens,
                max_length=max_seq_length,
                truncation=True,
                return_token_type_ids=True,
                padding="max_length",
                is_split_into_words=True
            )

            # 为每一个字符添加NER标签
            crf_labels = [0] * len(content_tokens)
            for ent in labels:
                s = int(ent["start_idx"])
                e = int(ent["end_idx"])
                e_type = ent["type"]
                if s < len(content_tokens) and e < len(content_tokens):
                    crf_labels[s] = ent2id['B+' + e_type]
                    for i in range(s + 1, e + 1):
                        crf_labels[i] = ent2id['I+' + e_type]

            crf_labels = [0] + [0] * len(query_tokens) + [0] + crf_labels + [0]
            crf_labels += [0] * (max_seq_length - len(crf_labels))

            # print(tokenizer.convert_ids_to_tokens(tokenized_examples["input_ids"]))
            # print(crf_labels)
            # print(labels)
            # print(query)
            # print("*"*100)

            tokenized_examples["labels"] = crf_labels
            tokenized_examples["offsets"] = len(query_tokens) + 2

            for k, v in tokenized_examples.items():
                all_tokenized_examples[k].append(v)
        return all_tokenized_examples

    def postprocess_predictions(
            self,
            examples,
            features,
            predictions: Tuple[np.ndarray, np.ndarray],
            output_dir: Optional[str] = None,
            prefix: Optional[str] = None,
    ):
        all_logits = predictions

        assert len(predictions) == len(
            features
        ), f"Got {len(predictions)} predictions and {len(features)} features."

        assert len(predictions) == len(
            examples
        ), f"Got {len(predictions)} predictions and {len(examples)} examples."

        all_predictions = []
        save_predictions = collections.defaultdict(list)
        for example_idx, example in enumerate(examples):
            logit = all_logits[example_idx]
            ent_type = example["ent_types"]
            offsets = features[example_idx]["offsets"]
            text = example["text"]
            example_key = example["example_ids"]

            pred_ents = self.decode(text, logit, ent_type, offsets)
            all_predictions.append(pred_ents)
            try:
                gold = [{
                    "start_idx": int(ent["start_idx"]),
                    "end_idx": int(ent["end_idx"]),
                    "type": ent["type"],
                    "entity": ent["entity"]
                } for ent in example["labels"]]

                save_predictions[example_key].append({
                    "ent_type": ent_type,
                    "mrc_query": example["querys"],
                    "text": text,
                    "pred": pred_ents,
                    "gold": gold,
                })
            except KeyError:
                save_predictions[example_key].append({
                    "ent_type": ent_type,
                    "mrc_query": example["querys"],
                    "text": text,
                    "pred": pred_ents,
                })

        if output_dir is not None:

            self.save_jsonl(os.path.join(output_dir, "tmp_predictions.json"), save_predictions)

            tmp_save_predictions = []
            for key, pred in save_predictions.items():
                text = pred[0]["text"]
                pred_ents = [ent for item in pred for ent in item["pred"]]
                try:
                    gold_ents = [ent for item in pred for ent in item["gold"]]
                except:
                    gold_ents = []
                tmp_save_predictions.append(
                    {
                        "text": text,
                        "pred": pred_ents,
                        "gold": gold_ents,
                        "new": [item for item in pred_ents if item not in gold_ents],
                        "lack": [item for item in gold_ents if item not in pred_ents]
                    }
                )

            assert os.path.isdir(output_dir), f"{output_dir} is not a directory."

            prediction_file = os.path.join(
                output_dir,
                "predictions.json" if prefix is None else f"{prefix}_predictions.json",
            )
            logger.info(f"Saving predictions to {prediction_file}.")
            self.save_jsonl(prediction_file, tmp_save_predictions)

        return all_predictions

    def decode(
            self,
            text: Optional[str] = None,
            pred_tokens: Tuple[np.ndarray, np.ndarray] = None,
            ent_type: Optional[str] = None,
            offset: Optional[int] = 0,
    ):
        pred_tokens = pred_tokens.tolist()
        bios = [self.id2ent[item] for item in pred_tokens if item != -1]
        bios = bios[offset:-1]  # 除去 query, CLS SEP token

        pred_ents = []
        start_index, end_index = -1, -1
        ent_type = None
        for indx, tag in enumerate(bios):
            if tag.startswith("B+"):
                if end_index != -1:
                    pred_ents.append(
                        {
                            "start_idx": start_index,
                            "end_idx": end_index,
                            "type": ent_type,
                            "entity": text[start_index:end_index + 1]
                        }
                    )
                # 新的实体
                start_index = indx
                end_index = indx
                ent_type = tag.split('+')[1]
                if indx == len(bios) - 1:
                    pred_ents.append(
                        {
                            "start_idx": start_index,
                            "end_idx": end_index,
                            "type": ent_type,
                            "entity": text[start_index:end_index + 1]
                        }
                    )
            elif tag.startswith('I+') and start_index != -1:
                _type = tag.split('+')[1]
                if _type == ent_type:
                    end_index = indx

                if indx == len(bios) - 1:
                    pred_ents.append(
                        {
                            "start_idx": start_index,
                            "end_idx": end_index,
                            "type": ent_type,
                            "entity": text[start_index:end_index + 1]
                        }
                    )
            else:
                if end_index != -1:
                    pred_ents.append(
                        {
                            "start_idx": start_index,
                            "end_idx": end_index,
                            "type": ent_type,
                            "entity": text[start_index:end_index + 1]
                        }
                    )
                start_index, end_index = -1, -1
                ent_type = None
        return pred_ents
