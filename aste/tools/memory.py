from typing import Callable, Optional, Dict

from ASTE.utils import config


class Memory:
    def __init__(self):
        self.best_epoch: int = 0
        self.opt_direction: str = 'min' if 'Loss' in config['model']['best-epoch-objective'].capitalize() else 'max'
        if self.opt_direction == 'min':
            self.func: Callable = min
            self.best_value: float = float('inf')
        elif self.opt_direction == 'max':
            self.func: Callable = max
            self.best_value: float = float('-inf')
        self.patience: Optional[int] = config['model']['early-stopping']
        self.early_stopping_objective: Optional[str] = None
        if self.patience is not None:
            self.early_stopping_objective = config['model']['best-epoch-objective']
            if 'Loss' in self.early_stopping_objective.capitalize():
                self.early_stopping_objective = 'Test loss'

    def reset(self):
        self.__init__()

    def update(self, epoch: int, values_dict: Dict) -> bool:
        value: float = values_dict[self.early_stopping_objective]
        improvement: bool = False
        best_loss = self.func(self.best_value, value)
        if best_loss != self.best_value:
            improvement = True
            self.best_epoch = epoch
            self.best_value = value
        return improvement

    def early_stopping(self, epoch: int) -> bool:
        if self.patience is None:
            return False
        return epoch - self.best_epoch > self.patience
