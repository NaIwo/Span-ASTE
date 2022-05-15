from transformers import BertModel
import torch

from ASTE.utils import config
from .base_embeddings import BaseEmbedding


class Bert(BaseEmbedding):
    def __init__(self, model_name: str = 'Bert embedding model'):
        dim: int = config['encoder']['bert']['embedding-dimension']
        super(Bert, self).__init__(embedding_dim=dim, model_name=model_name)
        self.model: BertModel = BertModel.from_pretrained(config['model']['bert']['source'])

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.model.forward(inputs, mask).last_hidden_state
