import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List
from abc import abstractmethod


class BaseAggregator:
    def __init__(self, input_dim: int, name: str = 'base aggregator', *args, **kwargs):
        self.name: str = name
        self.input_dim: int = input_dim

    def aggregate(self, embeddings: torch.Tensor, spans: List[torch.Tensor], *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_dim(self):
        raise NotImplementedError

    @staticmethod
    def pad_sequence(agg_embeddings: List[torch.Tensor]) -> torch.Tensor:
        return pad_sequence(agg_embeddings, padding_value=0, batch_first=True)
