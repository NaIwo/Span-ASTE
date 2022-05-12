import logging
from typing import List, Dict, Any

import torch

from ASTE.aste.models import ModelOutput, ModelLoss, ModelMetric
from ASTE.utils import config


class BaseModel(torch.nn.Module):
    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.performed_epochs: int = 0
        self.warmup: bool = False
        self.trainable: bool = True

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplemented

    def get_loss(self, model_out: ModelOutput) -> ModelLoss:
        raise NotImplemented

    def update_metrics(self, model_out: ModelOutput) -> None:
        raise NotImplemented

    def get_metrics_and_reset(self) -> ModelMetric:
        metrics: ModelMetric = self.get_metrics()
        self.reset_metrics()
        return metrics

    def get_metrics(self) -> ModelMetric:
        raise NotImplemented

    def reset_metrics(self) -> None:
        raise NotImplemented

    def update_trainable_parameters(self) -> None:
        self.performed_epochs += 1

    def freeze(self) -> None:
        logging.info(f"Model '{self.model_name}' freeze.")
        self.trainable = False
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        logging.info(f"Model '{self.model_name}' unfreeze.")
        self.trainable = True
        for param in self.parameters():
            param.requires_grad = True

    def get_params_and_lr(self) -> List[Dict]:
        return [{
            "param": self.parameters(), 'lr': config['model']['learning-rate']
        }]