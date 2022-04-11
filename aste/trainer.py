from ASTE.aste.models.base_model import BaseModel
from ASTE.utils import config
from ASTE.dataset.domain.const import ChunkCode
from .utils import ignore_index
from .losses import DiceLoss

import torch
from torchmetrics import FBetaScore, Accuracy, Precision, Recall, F1Score, MetricCollection
from torch.utils.data import DataLoader
from typing import Optional, Dict, Callable
import logging
from datetime import datetime
from tqdm import tqdm
import os


class Memory:
    def __init__(self, opt_direction: str = 'min'):
        self.best_epoch: int = 0
        self.opt_direction: str = opt_direction
        if self.opt_direction == 'min':
            self.func: Callable = min
            self.best_value: float = float('inf')
        elif self.opt_direction == 'max':
            self.func: Callable = max
            self.best_value: float = float('-inf')
        self.patience: Optional[int] = config['model']['early-stopping']
        self.early_stopping_objective: Optional[str] = None
        if self.patience is not None:
            self.early_stopping_objective = config['model']['early-stopping-objective'].capitalize()
            if 'Loss' in self.early_stopping_objective:
                self.early_stopping_objective = 'Test_loss'

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


class Trainer:
    def __init__(self, model: BaseModel, metrics: Optional[Metrics] = None, save_path: Optional[str] = None):
        logging.info(f"Model '{model.model_name}' has been initialized.")
        self.model: BaseModel = model.to(config['general']['device'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['model']['learning-rate'])
        self.chunk_loss = DiceLoss(ignore_index=int(ChunkCode.NOT_RELEVANT), alpha=0.99)
        self.memory = Memory('max')
        if metrics is None:
            self.metrics = Metrics(metrics=[
                Precision(num_classes=1, multiclass=False),
                Recall(num_classes=1, multiclass=False),
                Accuracy(num_classes=1, multiclass=False),
                FBetaScore(num_classes=1, multiclass=False, beta=0.5),
                F1Score(num_classes=1, multiclass=False)
            ]).to(config['general']['device'])
        else:
            self.metrics = metrics.to(config['general']['device'])

        if save_path is None:
            self.save_path = os.path.join(os.getcwd(), 'results', datetime.now().strftime("%Y%m%d-%H%M%S"),
                                          f'{self.model.model_name.replace(" ", "_").lower()}.pth')
        else:
            self.save_path = save_path

    def train(self, train_data: DataLoader, dev_data: Optional[DataLoader] = None) -> None:
        os.makedirs(self.save_path[:self.save_path.rfind(os.sep)], exist_ok=False)
        training_start_time: datetime.time = datetime.now()
        logging.info(f'Training start at time: {training_start_time}')
        for epoch in range(config['model']['epochs']):
            epoch_loss = self._training_epoch(train_data)
            logging.info(f"Epoch: {epoch + 1}/{config['model']['epochs']}. Epoch loss: {epoch_loss:.3f}")
            early_stopping: bool = self._eval(epoch=epoch, dev_data=dev_data)
            if early_stopping:
                logging.info(f'Early stopping performed. Patience factor: {self.memory.patience}')
                break

        training_stop_time: datetime.time = datetime.now()
        logging.info(f'Training stop at time: {training_stop_time}')
        logging.info(f'Training time in seconds: {(training_stop_time - training_start_time).seconds}')
        logging.info(f'Best epoch: {self.memory.best_epoch}')

    def _training_epoch(self, train_data: DataLoader) -> float:
        self.model.train()
        epoch_loss = 0.
        for batch_idx, batch in enumerate(bar := tqdm(train_data)):
            sentence, chunk_label, mask = batch
            model_out: torch.Tensor = self.model(sentence, mask)
            loss = self.chunk_loss(model_out.view([-1, model_out.shape[-1]]), chunk_label.view([-1]))
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            epoch_loss += loss
            bar.set_description(f'Loss: {epoch_loss / (batch_idx + 1):.3f}  ')
        return epoch_loss / len(train_data)

    def _eval(self, epoch: int, dev_data: DataLoader) -> bool:
        if dev_data is not None:
            values_dict: Dict = self.test(dev_data)
            improvement: bool = self.memory.update(epoch, values_dict)
            if improvement:
                logging.info(f'Improvement has occurred. Saving the model in the path: {self.save_path}')
                self.save_model(self.save_path)
            if self.memory.early_stopping(epoch):
                return True
        return False

    def test(self, test_data: DataLoader) -> Dict:
        self.model.eval()
        return_values: Dict = {}
        logging.info(f'Test started...')
        test_loss: float = 0.
        with torch.no_grad():
            for batch in tqdm(test_data):
                sentence, chunk_label, mask = batch
                model_out: torch.Tensor = self.model(sentence, mask)
                chunk_label = chunk_label.view([-1])
                model_out = model_out.view([-1, model_out.shape[-1]])
                test_loss += self.chunk_loss(model_out, chunk_label)
                self.metrics(model_out, chunk_label)
            logging.info(f'Test loss: {test_loss / len(test_data):.3f}')
            metric_results: Dict = self.metrics.compute()
            return_values.update(metric_results)
            self.metrics.reset()
        return_values['Test_loss'] = test_loss / len(test_data)
        return return_values

    def save_model(self, save_path: str) -> None:
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, save_path: str) -> None:
        self.model.load_state_dict(torch.load(save_path))

    def predict(self, sentence, mask) -> torch.Tensor:
        self.model.eval()
        return self.model(sentence, mask)
