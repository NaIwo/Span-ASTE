import logging
from typing import Optional, Dict, List

import torch
from torchmetrics import MetricCollection
from torchmetrics import FBetaScore, Accuracy, Precision, Recall, F1Score

from ASTE.aste.utils import ignore_index
from ASTE.dataset.domain.const import ChunkCode


class Metrics(MetricCollection):
    def __init__(self, name: str, ignore_index: Optional[int] = int(ChunkCode.NOT_RELEVANT), *args, **kwargs):
        self.ignore_index = ignore_index
        self.name: str = name
        super().__init__(*args, **kwargs)

    @ignore_index
    def forward(self, preds, target):
        super(Metrics, self).forward(preds, target)

    def compute(self):
        logging.info(f'Metrics: ')
        logging.info(f'\t--- {self.name} ---')
        metric_name: str
        score: torch.Tensor
        computed: Dict = super(Metrics, self).compute()
        for metric_name, score in computed.items():
            logging.info(f'\t->\t{metric_name}: {score.item()}')
        return computed


selected_metrics: List = [
    Precision(num_classes=1, multiclass=False),
    Recall(num_classes=1, multiclass=False),
    Accuracy(num_classes=1, multiclass=False),
    FBetaScore(num_classes=1, multiclass=False, beta=0.5),
    F1Score(num_classes=1, multiclass=False)
]
