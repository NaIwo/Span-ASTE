from typing import List

import torch
from torch import Tensor
from torch.nn import Module

from .base_agg import BaseAggregator


class SumAggregator(BaseAggregator, Module):
    def __init__(self, input_dim: int, model_name: str = 'Sum Aggregator', *args, **kwargs):
        Module.__init__(self)
        BaseAggregator.__init__(self, input_dim=input_dim, model_name=model_name)
        self._out_dim: int = input_dim

    @property
    def output_dim(self):
        return self._out_dim

    def _get_agg_sentence_embeddings(self, sentence_embeddings: Tensor, sentence_spans: Tensor) -> Tensor:
        sentence_agg_embeddings: List = list()
        span: Tensor
        for span in sentence_spans:
            span_emb: Tensor = sentence_embeddings[span[0]:span[1] + 1]
            agg_emb: Tensor = torch.sum(span_emb, dim=0)
            sentence_agg_embeddings.append(agg_emb)
        return torch.stack(sentence_agg_embeddings, dim=0)
