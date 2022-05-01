import torch
from typing import List, Dict, TypeVar
from ASTE.aste.tools.metrics import Metrics

M = TypeVar('M', bound='ModelLoss')


class ModelOutput:
    def __init__(self, chunker_output: torch.Tensor, predicted_spans: List[torch.Tensor]):
        self.chunker_output: torch.Tensor = chunker_output
        self.predicted_spans: List[torch.Tensor] = predicted_spans

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str({
            'chunker_output': self.chunker_output,
            'predicted_spans': self.predicted_spans
        })


class ModelLoss:
    def __init__(self, chunker_loss: torch.Tensor = 0.):
        self.chunker_loss: torch.Tensor = chunker_loss

    def backward(self) -> None:
        self.full_loss.backward()

    @property
    def full_loss(self) -> torch.Tensor:
        return self.chunker_loss

    @property
    def _loss_dict(self) -> Dict:
        return {
            'chunker_loss': float(self.chunker_loss),
            'full_loss': float(self.full_loss)
        }

    def __radd__(self, other) -> torch.Tensor:
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

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str({name: round(value, 4) for name, value in self._loss_dict.items()})


class ModelMetric:
    def __init__(self, chunker_metric: Dict):
        self.chunker_metric: Dict = chunker_metric

    @property
    def _all_metrics(self) -> Dict:
        return {
            'chunker metrics': self.chunker_metric
        }

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self._all_metrics)

    def __iter__(self):
        for metrics in self._all_metrics:
            yield metrics

