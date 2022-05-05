import torch
from typing import Any


class BaseEmbedding(torch.nn.Module):
    def __init__(self, model_name: str = 'Base embedding model'):
        super(BaseEmbedding, self).__init__()
        self.model: Any = None
        self.model_name: str = model_name
        self.fully_trainable: bool = True

    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplemented
