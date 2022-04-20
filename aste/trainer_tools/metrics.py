import logging
from typing import Optional, Dict

import torch
from torchmetrics import MetricCollection

from ASTE.aste.utils import ignore_index
from ASTE.dataset.domain.const import ChunkCode


class Metrics(MetricCollection):
    def __init__(self, ignore_index: Optional[int] = int(ChunkCode.NOT_RELEVANT), *args, **kwargs):
        self.ignore_index = ignore_index
        super().__init__(*args, **kwargs)

    @ignore_index
    def forward(self, preds, target):
        super(Metrics, self).forward(preds, target)

    def compute(self):
        logging.info(f'Metrics: ')
        metric_name: str
        score: torch.Tensor
        computed: Dict = super(Metrics, self).compute()
        for metric_name, score in computed.items():
            logging.info(f'\t->\t{metric_name}: {score.item()}')
        return computed
