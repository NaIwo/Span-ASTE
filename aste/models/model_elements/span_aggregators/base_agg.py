import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List


class BaseAggregator:
    def __init__(self, embeddings_dim: int, name: str = 'base aggregator'):
        self.name: str = name
        self.embeddings_dim: int = embeddings_dim

    def aggregate(self, embeddings: torch.Tensor, spans: List[torch.Tensor], *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def pad_sequence(agg_embeddings: List[torch.Tensor]) -> torch.Tensor:
        return pad_sequence(agg_embeddings, padding_value=0, batch_first=True)
