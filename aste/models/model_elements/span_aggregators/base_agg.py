from abc import abstractmethod
from typing import List

from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


class BaseAggregator:
    def __init__(self, input_dim: int, model_name: str = 'base aggregator', *args, **kwargs):
        self.model_name: str = model_name
        self.input_dim: int = input_dim

    def aggregate(self, embeddings: Tensor, spans: List[Tensor]) -> Tensor:
        agg_embeddings: List = list()
        sentence_embeddings: Tensor
        sentence_spans: Tensor
        for sentence_embeddings, sentence_spans in zip(embeddings, spans):
            sentence_agg_embeddings: Tensor = self._get_agg_sentence_embeddings(sentence_embeddings, sentence_spans)
            agg_embeddings.append(sentence_agg_embeddings)

        return self.pad_sequence(agg_embeddings)

    def _get_agg_sentence_embeddings(self, sentence_embeddings: Tensor, sentence_spans: Tensor) -> Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_dim(self):
        raise NotImplementedError

    @staticmethod
    def pad_sequence(agg_embeddings: List[Tensor]) -> Tensor:
        return pad_sequence(agg_embeddings, padding_value=0., batch_first=True)
