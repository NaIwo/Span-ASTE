from typing import List

import torch
from torch import Tensor

from .base_agg import BaseAggregator


class LastElementAggregator(BaseAggregator):
    def __init__(self, input_dim: int, model_name: str = 'Last Element Aggregator', *args, **kwargs):
        self._out_dim: int = input_dim
        BaseAggregator.__init__(self, input_dim=self._out_dim, model_name=model_name)

    @property
    def output_dim(self):
        return self._out_dim

    def _get_agg_sentence_embeddings(self, sentence_embeddings: Tensor, sentence_spans: Tensor) -> Tensor:
        sentence_agg_embeddings: List = list()
        span: Tensor
        for span in sentence_spans:
            sentence_agg_embeddings.append(sentence_embeddings[span[1]])
        return torch.stack(sentence_agg_embeddings, dim=0)
