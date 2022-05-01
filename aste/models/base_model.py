import torch
from typing import Any


class BaseModel(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplemented
