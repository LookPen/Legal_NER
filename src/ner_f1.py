# Copyright 2020 The HuggingFace Datasets Authors.
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
""" NER F1 metric. """
import os
import datasets
from datasets import load_metric as load_hf_metric

_CITATION = """\
    Nothing
"""

_DESCRIPTION = """
    Nothing
"""


class NERF1(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            features=datasets.Features(
                {
                    "predictions":
                        datasets.Sequence(
                            {
                                "entity": datasets.Value("string"),
                                "type": datasets.Value("string"),
                                "start_idx": datasets.Value("string"),
                                "end_idx": datasets.Value("string")
                            }
                        )
                    ,
                    "references":
                        datasets.Sequence(
                            {
                                "entity": datasets.Value("string"),
                                "type": datasets.Value("string"),
                                "start_idx": datasets.Value("string"),
                                "end_idx": datasets.Value("string")
                            }
                        )
                    ,
                }
            ),
        )

    def _compute(self, predictions, references):
        assert len(predictions) == len(references), print("the length of predictions and references is not equal!!")

        '''
            注意：这里传入的predictions的格式是类似于表格的形式，比如：
            pred:
            {
                "start_idx": [10, 10],
                "end_idx": [14, 12],
                "type": ["疾病", "器官"],
                "entity": ["口腔溃疡", "口腔"]
            }
        '''

        predictions = [
            [(s, e, t, ent) for s, e, t, ent in zip(pred["start_idx"], pred["end_idx"], pred["type"], pred["entity"])]
            for pred in predictions]
        references = [
            [(s, e, t, ent) for s, e, t, ent in zip(gold["start_idx"], gold["end_idx"], gold["type"], gold["entity"])]
            for gold in references]

        num_common = 0
        num_pred = 1e-9
        num_gold = 1e-9
        for pred, gold in zip(predictions, references):

            pred = set(pred)
            gold = set(gold)

            num_common += len(pred & gold)
            num_pred += len(pred)
            num_gold += len(gold)

        f1, precision, recall = 2 * num_common / (num_pred + num_gold), num_common / num_pred, num_common / num_gold

        return {"f1": f1,
                "precision": precision,
                "recall": recall}


def load_ner_metric(metric_path_or_name=None, cache_dir=None):
    def compute_metrics(predictions, eval_examples):
        references = [
            example["labels"]
            for example in eval_examples
        ]

        predictions = [
            [
                {
                    "entity": ent["entity"],
                    "type": ent["type"],
                    "start_idx": str(ent["start_idx"]),
                    "end_idx": str(ent["end_idx"])
                } for ent in pred
            ]
            for pred in predictions
        ]

        results = metric.compute(predictions=predictions, references=references)
        return results


    # current_path = os.path.abspath(__file__)
    # current_file_name = current_path.split("/")[-1]
    # ner_f1_file_name = "ner_f1.py"
    # ner_f1_path = current_path.replace(current_file_name, ner_f1_file_name)
    ner_f1_path = r"D:\Source\promptNER\src\ner_f1.py"
    metric = load_hf_metric(ner_f1_path, cache_dir=cache_dir)

    return compute_metrics
