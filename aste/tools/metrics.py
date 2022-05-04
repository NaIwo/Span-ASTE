from typing import Optional, Dict, List

from torchmetrics import MetricCollection
from torchmetrics import FBetaScore, Accuracy, Precision, Recall, F1Score

from ASTE.aste.utils import ignore_index
from ASTE.dataset.domain.const import ChunkCode


class Metrics(MetricCollection):
    def __init__(self, name: str, ignore_index: Optional[int] = ChunkCode.NOT_RELEVANT, *args, **kwargs):
        self.ignore_index = ignore_index
        self.name: str = name
        super().__init__(*args, **kwargs)

    @ignore_index
    def forward(self, preds, target):
        super(Metrics, self).forward(preds, target)

    def compute(self):
        computed: Dict = super(Metrics, self).compute()
        for metric_name, score in computed.items():
            computed[metric_name] = float(score)
        return computed


def get_selected_metrics(num_classes: int = 1, multiclass: bool = False) -> List:
    return [
        Precision(num_classes=num_classes, multiclass=multiclass),
        Recall(num_classes=num_classes, multiclass=multiclass),
        Accuracy(num_classes=num_classes, multiclass=multiclass),
        FBetaScore(num_classes=num_classes, multiclass=multiclass, beta=0.5),
        F1Score(num_classes=num_classes, multiclass=multiclass)
    ]
