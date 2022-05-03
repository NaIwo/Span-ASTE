import torch
from torch import Tensor
from typing import List, Dict, TypeVar

from ASTE.dataset.reader import Batch

M = TypeVar('M', bound='ModelLoss')


class ModelOutput:
    NAME: str = 'Outputs'

    def __init__(self, batch: Batch, chunker_output: Tensor, predicted_spans: List[Tensor], triplet_results: Tensor):
        self.batch: Batch = batch
        self.chunker_output: Tensor = chunker_output
        self.predicted_spans: List[Tensor] = predicted_spans
        self.triplet_results: Tensor = triplet_results

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

    def __init__(self, *, chunker_loss: Tensor = torch.tensor([0.])):
        self.chunker_loss: Tensor = chunker_loss

    def backward(self) -> None:
        self.full_loss.backward()

    @property
    def full_loss(self) -> Tensor:
        return self.chunker_loss

    @property
    def _loss_dict(self) -> Dict:
        return {
            'chunker_loss': float(self.chunker_loss),
            'full_loss': float(self.full_loss)
        }

    def __radd__(self, other) -> Tensor:
        return self.__add__(other)

    def __add__(self, other: M) -> M:
        return ModelLoss(
            chunker_loss=self.chunker_loss + other.chunker_loss
        )

    def __truediv__(self, other: int) -> M:
        return ModelLoss(
            chunker_loss=self.chunker_loss / other
        )

    def __iter__(self):
        for element in self._loss_dict.items():
            yield element

    def __str__(self):
        return str({name: round(value, 4) for name, value in self._loss_dict.items()})

    @property
    def logs(self) -> Dict:
        return self._loss_dict


class ModelMetric:
    NAME: str = 'Metrics'

    def __init__(self, chunker_metric: Dict):
        self.chunker_metric: Dict = chunker_metric

    @property
    def _all_metrics(self) -> Dict:
        return {
            'chunker_metrics': self.chunker_metric
        }

    def __str__(self):
        return str(self._all_metrics)

    def __iter__(self):
        for metrics in self._all_metrics:
            yield metrics

    @property
    def logs(self) -> Dict:
        return self._all_metrics
