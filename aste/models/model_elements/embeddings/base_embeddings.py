import torch
from typing import Any
import logging


class BaseEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim: int, model_name: str = 'Base embedding model'):
        super(BaseEmbedding, self).__init__()
        self.model: Any = None
        self.model_name: str = model_name
        self.fully_trainable: bool = True
        self.embedding_dim: int = embedding_dim

    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplemented

    def freeze(self) -> None:
        logging.info(f"Model '{self.model_name}' freeze.")
        if not self.fully_trainable:
            return
        for param in self.parameters():
            param.requires_grad = False
        self.fully_trainable = False

    def unfreeze(self) -> None:
        logging.info(f"Model '{self.model_name}' unfreeze.")
        if self.fully_trainable:
            return
        for param in self.parameters():
            param.requires_grad = True
        self.fully_trainable = True
