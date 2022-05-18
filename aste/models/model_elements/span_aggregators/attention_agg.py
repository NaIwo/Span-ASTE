import torch
from torch.nn import Module
from typing import List

from .base_agg import BaseAggregator


class AttentionAggregator(BaseAggregator, Module):
    def __init__(self, input_dim: int, model_name: str = 'Attention Aggregator', *args, **kwargs):
        Module.__init__(self)
        BaseAggregator.__init__(self, input_dim=input_dim, model_name=model_name)
        self._out_dim: int = input_dim
        self.query_linear = torch.nn.Linear(input_dim, input_dim)
        self.softmax = torch.nn.Softmax(dim=0)

    @property
    def output_dim(self):
        return self._out_dim

    def aggregate(self, embeddings: torch.Tensor, spans: List[torch.Tensor], *args, **kwargs) -> torch.Tensor:
        agg_embeddings: List = list()
        sentence_embeddings: torch.Tensor
        sentence_spans: torch.Tensor
        for sentence_embeddings, sentence_spans in zip(embeddings, spans):
            sentence_agg_embeddings = self._get_agg_sentence_embeddings(sentence_embeddings, sentence_spans)
            agg_embeddings.append(torch.stack(sentence_agg_embeddings, dim=0))
        return self.pad_sequence(agg_embeddings)

    def _get_agg_sentence_embeddings(self, sentence_embeddings: torch.Tensor, sentence_spans: torch.Tensor) -> List:
        sentence_agg_embeddings: List = list()
        span: torch.Tensor
        for span in sentence_spans:
            span_emb: torch.Tensor = sentence_embeddings[span[0]:span[1]+1]
            attention: torch.Tensor = self._attention(span_emb)
            agg_emb: torch.Tensor = torch.sum(attention, dim=0)
            sentence_agg_embeddings.append(agg_emb)
        return sentence_agg_embeddings

    def _attention(self, span_emb: torch.Tensor) -> torch.Tensor:
        query: torch.Tensor = self.query_linear(span_emb)
        value: torch.Tensor = span_emb.permute(1, 0)

        weights: torch.Tensor = query @ value
        weights = self.softmax(weights)

        attention: torch.Tensor = weights @ span_emb
        return attention
