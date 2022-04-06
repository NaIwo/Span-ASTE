from .base_model import BaseModel
from ASTE.utils import config
from ASTE.aste.model_elements.embeddings.bert_embeddings import Bert


import torch


class BertBaseModel(BaseModel):
    def __init__(self, model_name='Bert Base Model'):
        super(BertBaseModel, self).__init__(model_name)
        self.embeddings_layer: Bert = Bert()
        self.dropout = torch.nn.Dropout(0.1)
        self.linear_layer_1 = torch.nn.Linear(config['encoder']['bert']['embedding-dimension'], 100)
        self.linear_layer_2 = torch.nn.Linear(100, 10)
        self.final_layer = torch.nn.Linear(10, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, sentence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        data: torch.Tensor = self.embeddings_layer(sentence, mask)
        layer: torch.nn.Linear
        for layer in [self.linear_layer_1, self.linear_layer_2]:
            data = layer(data)
            data = self.dropout(data)
        data = self.final_layer(data)
        return self.sigmoid(data)
