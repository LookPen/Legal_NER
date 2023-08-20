import torch
import torch.nn as nn
from transformers.modeling_utils import  PreTrainedModel
from transformers import AutoModel, BertPreTrainedModel
from transformers.modeling_outputs import  ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple
from src.crf import CRF

@dataclass
class NERModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    seq_logtis: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.FloatTensor] = None
    attentions: Optional[torch.FloatTensor] = None


class PromptNER(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.bert = AutoModel.from_config(config)

        hidden_dim = self.config.hidden_size

        crf_hidden_size = 128
        if hasattr(config, "num_ner_labels"):
            num_ner_labels = config.num_ner_labels
        else:
            raise ValueError("需要提供ner labels")

        self.crf_fc = nn.Sequential(
            nn.Linear(hidden_dim, crf_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(crf_hidden_size, num_ner_labels)
        )
        self.crf_module = CRF(num_tags=num_ner_labels, batch_first=True)

        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self,
                input_ids,
                attention_mask,
                labels
                ):
        ptm_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # 0816 batch_size：per_device_train_batch_size设置
        # seq_output: [batch_size, seq_len, bert_dim]
        seq_output, _ = ptm_output[0], ptm_output[1]
        # [batch_size, seq_len, num_ner_labels]
        emissions = self.crf_fc(seq_output)
        total_loss = None
        crf_decode_seqs = None
        if labels is not None:
            # TODO 0816 这个损失函数表示的是啥，损失函数为啥取反，crf.crf_module.forward
            total_loss = -1 * self.crf_module(
                emissions=emissions,
                tags=labels.long(),
                mask=attention_mask.byte(),
                reduction="mean"
            )

        if not self.training:
            seq_length = input_ids.size(1)
            crf_decode_seqs = self.crf_module.decode(emissions=emissions,
                                                     mask=attention_mask.byte()
                                                     )
            for line in crf_decode_seqs:
                padding = [-1] * (seq_length - len(line))
                line += padding

            crf_decode_seqs = torch.tensor(crf_decode_seqs).to(input_ids.device)

        # 0820 optimizer、反向传播 等再Trainer中封装好了，这里必须返回ModelOutput类型
        return NERModelOutput(
            loss=total_loss,
            seq_logtis=crf_decode_seqs
        )