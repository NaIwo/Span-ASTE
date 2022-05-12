import torch
from torch.nn import Module
from typing import List

from .base_agg import BaseAggregator
from ASTE.utils import config


class EndPointAggregator(BaseAggregator, Module):
    def __init__(self, input_dim: int, model_name: str = 'End Point Aggregator', *args, **kwargs):
        Module.__init__(self)

        distance_embedding_dim: int = config['model']['aggregators']['endpoint']['distance-embedding-dim']
        self.distance_embedding = torch.nn.Linear(1, distance_embedding_dim)

        self._out_dim: int = 2 * input_dim + distance_embedding_dim
        BaseAggregator.__init__(self, input_dim=self._out_dim, model_name=model_name)

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
            distance = self._get_distance_embedding(span)
            sentence_agg_embeddings.append(
                torch.cat((sentence_embeddings[span[0]], sentence_embeddings[span[1]], distance))
            )
        return sentence_agg_embeddings

    def _get_distance_embedding(self, span: torch.Tensor) -> torch.Tensor:
        distance: torch.Tensor = (span[1] - span[0]).type(torch.float).unsqueeze(0)
        distance = self.distance_embedding(distance)
        distance = torch.tanh(distance)
        return distance
