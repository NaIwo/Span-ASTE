import torch
from torch.nn import Module
from typing import List

from ASTE.utils import config
from .base_agg import BaseAggregator


class AttentionAggregator(BaseAggregator, Module):
    def __init__(self, input_dim: int, model_name: str = 'Attention Aggregator', *args, **kwargs):
        Module.__init__(self)
        BaseAggregator.__init__(self, input_dim=input_dim, model_name=model_name)
        self._out_dim: int = input_dim

        self.linear_layer_1 = torch.nn.Linear(input_dim, 500)
        self.linear_layer_2 = torch.nn.Linear(500, 300)
        self.linear_layer_3 = torch.nn.Linear(300, 100)
        self.final_layer = torch.nn.Linear(100, 1)
        self.dropout = torch.nn.Dropout(0.1)
        self.batch_norm = torch.nn.BatchNorm1d(input_dim)
        self.softmax = torch.nn.Softmax(dim=-1)

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
            attention = self._get_attention_weights(span_emb)
            agg_emb: torch.Tensor = torch.sum(attention[..., None] * span_emb, dim=0)
            sentence_agg_embeddings.append(agg_emb)
        return sentence_agg_embeddings

    def _get_attention_weights(self, span_emb: torch.Tensor) -> torch.Tensor:
        if span_emb.shape[0] == 1:
            return torch.tensor(1., device=config['general']['device'])
        attention: torch.Tensor = span_emb.unsqueeze(0)
        attention = self.batch_norm(torch.permute(attention, (0, 2, 1)))
        attention = torch.permute(attention, (0, 2, 1))

        layer: torch.nn.Linear
        for layer in [self.linear_layer_1, self.linear_layer_2, self.linear_layer_3]:
            attention = layer(attention)
            attention = torch.relu(attention)
            attention = self.dropout(attention)

        attention = self.final_layer(attention)
        return self.softmax(attention.squeeze())
