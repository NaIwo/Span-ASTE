import torch
from torch import Tensor
from typing import List, Dict, TypeVar, Optional

from ASTE.utils import config
from ASTE.dataset.reader import Batch

ML = TypeVar('ML', bound='ModelLoss')
MM = TypeVar('MM', bound='ModelMetric')


class ModelOutput:
    NAME: str = 'Outputs'

    def __init__(self, batch: Batch, chunker_output: Tensor, predicted_spans: List[Tensor], triplet_results: Tensor):
        self.batch: Batch = batch
        self.chunker_output: Tensor = chunker_output
        self.predicted_spans: List[Tensor] = predicted_spans
        self.triplet_results: Tensor = triplet_results

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self):
        return str({
            'batch': [sample.sentence_obj[0].sentence for sample in self.batch],
            'chunker_output': self.chunker_output.shape,
            'predicted_spans': f'Predicted spans num: {[spans.shape[0] for spans in self.predicted_spans]}',
            'triplet_results': f'Prediction shape: {self.triplet_results.shape}. '
                               f'Number of meaningful values: {int(torch.sum(self.triplet_results > 0))}'
        })


class ModelLoss:
    NAME: str = 'Losses'

    def __init__(self, *, chunker_loss: Tensor = 0., triplet_loss: Tensor = 0., weighted: bool = True):
        self.chunker_loss: Tensor = chunker_loss
        self.triplet_loss: Tensor = triplet_loss

        if weighted:
            self._include_weights()

    @classmethod
    def from_instances(cls, *, chunker_loss: ML, triplet_loss: ML, weighted: bool = False) -> ML:
        return cls(
            chunker_loss=chunker_loss.chunker_loss,
            triplet_loss=triplet_loss.triplet_loss,
            weighted=weighted
        )

    def _include_weights(self) -> None:
        self.chunker_loss *= config['model']['chunker']['loss-weight']
        self.triplet_loss *= config['model']['triplet-extractor']['loss-weight']

    def backward(self) -> None:
        self.full_loss.backward()

    @property
    def full_loss(self) -> Tensor:
        return self.chunker_loss + self.triplet_loss

    @property
    def _loss_dict(self) -> Dict:
        return {
            'chunker_loss': float(self.chunker_loss),
            'triplet_loss': float(self.triplet_loss),
            'full_loss': float(self.full_loss)
        }

    def __radd__(self, other: ML) -> ML:
        return self.__add__(other)

    def __add__(self, other: ML) -> ML:
        return ModelLoss(
            chunker_loss=self.chunker_loss + other.chunker_loss,
            triplet_loss=self.triplet_loss + other.triplet_loss,
            weighted=False
        )

    def __truediv__(self, other: int) -> ML:
        return ModelLoss(
            chunker_loss=self.chunker_loss / other,
            triplet_loss=self.triplet_loss / other,
            weighted=False
        )

    def __iter__(self):
        for element in self._loss_dict.items():
            yield element

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self):
        return str({name: round(value, 4) for name, value in self._loss_dict.items()})

    @property
    def logs(self) -> Dict:
        return self._loss_dict


class ModelMetric:
    NAME: str = 'Metrics'

    def __init__(self, *, chunker_metric: Optional[Dict] = None, triplet_metric: Optional[Dict] = None,
                 independent_matrix_metric: Optional[Dict] = None):
        self.chunker_metric: Optional[Dict] = chunker_metric
        self.triplet_metric: Optional[Dict] = triplet_metric
        self.independent_matrix_metric: Optional[Dict] = independent_matrix_metric

    @classmethod
    def from_instances(cls, *, chunker_metric: MM, triplet_metric: MM) -> MM:
        return cls(
            chunker_metric=chunker_metric.chunker_metric,
            independent_matrix_metric=triplet_metric.independent_matrix_metric,
            triplet_metric=triplet_metric.triplet_metric
        )

    @property
    def _all_metrics(self) -> Dict:
        return {
            'chunker_metrics': self.chunker_metric,
            'triplet_metric': self.triplet_metric,
            'independent_matrix_metric': self.independent_matrix_metric,
        }

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self):
        return str(self._all_metrics)

    def __iter__(self):
        for metrics in self._all_metrics:
            yield metrics

    @property
    def logs(self) -> Dict:
        return self._all_metrics
