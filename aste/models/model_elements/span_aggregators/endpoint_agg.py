from typing import List

import torch
from torch import Tensor
from torch.nn import Module

from .base_agg import BaseAggregator


class EndPointAggregator(BaseAggregator, Module):
    def __init__(self, input_dim: int, model_name: str = 'End Point Aggregator', *args, **kwargs):
        Module.__init__(self)

        distance_embedding_dim: int = 2
        self.distance_embedding = torch.nn.Linear(1, distance_embedding_dim)

        self._out_dim: int = 2 * input_dim + distance_embedding_dim
        BaseAggregator.__init__(self, input_dim=self._out_dim, model_name=model_name)

    @property
    def output_dim(self):
        return self._out_dim

    def _get_agg_sentence_embeddings(self, sentence_embeddings: Tensor, sentence_spans: Tensor) -> Tensor:
        sentence_agg_embeddings: List = list()
        span: Tensor
        for span in sentence_spans:
            distance = self._get_distance_embedding(span)
            sentence_agg_embeddings.append(
                torch.cat((sentence_embeddings[span[0]], sentence_embeddings[span[1]], distance))
            )
        return torch.stack(sentence_agg_embeddings, dim=0)

    def _get_distance_embedding(self, span: Tensor) -> Tensor:
        distance: Tensor = (span[1] - span[0]).type(torch.float).unsqueeze(0)
        distance = self.distance_embedding(distance)
        distance = torch.tanh(distance)
        return distance
