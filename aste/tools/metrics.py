from typing import Optional, Dict, List
import torch
from torch import Tensor

from torchmetrics import MetricCollection
from torchmetrics import Metric as TorchMetric
from torchmetrics import FBetaScore, Accuracy, Precision, Recall, F1Score

from ASTE.aste.utils import ignore_index


class Metric(MetricCollection):
    def __init__(self, name: str, ignore_index: Optional[int] = None, *args, **kwargs):
        self.ignore_index = ignore_index
        self.name: str = name
        super().__init__(*args, **kwargs)

    @ignore_index
    def forward(self, preds, target):
        super(Metric, self).forward(preds, target)

    def compute(self):
        computed: Dict = super(Metric, self).compute()
        for metric_name, score in computed.items():
            computed[metric_name] = float(score)
        return computed


class TripletMetric(TorchMetric):
    def __init__(self, dist_sync_on_step: bool = False):
        TorchMetric.__init__(self, dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_predicted", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_target", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds = preds.unique(dim=0)
        target = target.unique(dim=0)

        self.correct += self._count_correct_num(preds, target)
        self.total_predicted += preds.numel()
        self.total_target += target.numel()

    @staticmethod
    def _count_correct_num(preds: Tensor, target: Tensor) -> int:
        all_data: Tensor = torch.cat([preds, target])
        uniques, counts = torch.unique(all_data, return_counts=True, dim=0)
        return uniques[counts == 2].shape[0]

    def compute(self) -> float:
        raise NotImplemented


class TripletPrecision(TripletMetric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

    def compute(self) -> float:
        if self.total_predicted == 0:
            return 0.
        return self.correct.float() / self.total_predicted


class TripletRecall(TripletMetric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

    def compute(self) -> float:
        if self.total_target == 0:
            return 0.
        return self.correct.float() / self.total_target


class TripletF1(TripletMetric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

    def compute(self) -> float:
        if self.total_predicted == 0 or self.total_target == 0:
            return 0.

        precision: float = self.correct.float() / self.total_predicted
        recall: float = self.correct.float() / self.total_target

        return 2 * (precision * recall) / (precision + recall)


def get_selected_metrics(num_classes: int = 1, multiclass: Optional[bool] = None, for_triplets: bool = False) -> List:
    if for_triplets:
        return [
            TripletPrecision(),
            TripletRecall(),
            TripletF1()
        ]
    else:
        return [
            Precision(num_classes=num_classes, multiclass=multiclass),
            Recall(num_classes=num_classes, multiclass=multiclass),
            Accuracy(num_classes=num_classes, multiclass=multiclass),
            FBetaScore(num_classes=num_classes, multiclass=multiclass, beta=0.5),
            F1Score(num_classes=num_classes, multiclass=multiclass)
        ]
