import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List
from abc import abstractmethod


class BaseAggregator:
    def __init__(self, input_dim: int, model_name: str = 'base aggregator', *args, **kwargs):
        self.model_name: str = model_name
        self.input_dim: int = input_dim

    def aggregate(self, embeddings: torch.Tensor, spans: List[torch.Tensor]) -> torch.Tensor:
        agg_embeddings: List = list()
        sentence_embeddings: torch.Tensor
        sentence_spans: torch.Tensor
        for sentence_embeddings, sentence_spans in zip(embeddings, spans):
            sentence_agg_embeddings = self._get_agg_sentence_embeddings(sentence_embeddings, sentence_spans)
            agg_embeddings.append(torch.stack(sentence_agg_embeddings, dim=0))
        return self.pad_sequence(agg_embeddings)

    def _get_agg_sentence_embeddings(self, sentence_embeddings: torch.Tensor, sentence_spans: torch.Tensor) -> List:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_dim(self):
        raise NotImplementedError

    @staticmethod
    def pad_sequence(agg_embeddings: List[torch.Tensor]) -> torch.Tensor:
        return pad_sequence(agg_embeddings, padding_value=0, batch_first=True)
