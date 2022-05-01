import torch
from torch.nn import Module
from typing import List

from .base_agg import BaseAggregator


class EndPointAggregator(BaseAggregator, Module):
    def __init__(self, name: str = 'End Point Aggregator'):
        BaseAggregator.__init__(self, name=name)
        Module.__init__(self)
        self.distance_embedding = torch.nn.Linear(1, 3)

    def aggregate(self, embeddings: torch.Tensor, spans: List[torch.Tensor], *args, **kwargs) -> torch.Tensor:
        agg_embeddings: List = list()
        sentence_embeddings: torch.Tensor
        sentence_spans: torch.Tensor
        for sentence_embeddings, sentence_spans in zip(embeddings, spans):
            sentence_agg_embeddings: List = list()
            span: torch.Tensor
            for span in sentence_spans:
                distance: torch.Tensor = (span[1] - span[0]).type(torch.float).unsqueeze(0)
                distance = self.distance_embedding(distance)
                distance = torch.tanh(distance)
                sentence_agg_embeddings.append(
                    torch.cat((sentence_embeddings[span[0]], sentence_embeddings[span[1]], distance))
                )
            agg_embeddings.append(torch.stack(sentence_agg_embeddings, dim=0))
        return self.pad_sequence(agg_embeddings)


