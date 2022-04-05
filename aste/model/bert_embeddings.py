from transformers import BertModel
import torch

from ASTE.utils import config


class Bert(torch.nn.Module):
    def __init__(self):
        super(Bert, self).__init__()
        self.bert: BertModel = BertModel.from_pretrained(config['encoder']['bert']['source'])

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.bert.forward(inputs, mask).last_hidden_state
