from transformers import BertModel
import torch

from ASTE.utils import config
from .base_embeddings import BaseEmbedding


class Bert(BaseEmbedding):
    def __init__(self):
        super(Bert, self).__init__(model_name='Bert embedding model')
        self.model: BertModel = BertModel.from_pretrained(config['encoder']['bert']['source'])

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.model.forward(inputs, mask).last_hidden_state
