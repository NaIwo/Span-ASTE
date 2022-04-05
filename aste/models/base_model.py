import torch


class BaseModel(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

    def forward(self, sentence: torch.Tensor, mask: torch.Tensor):
        raise NotImplemented
