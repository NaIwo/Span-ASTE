from ASTE.aste.models.base_model import BaseModel
from ASTE.utils import config
from ASTE.dataset.domain.const import ChunkCode
from .utils import ignore_index
from .losses import DiceLoss

import torch
from torchmetrics import FBetaScore, Accuracy, Precision, Recall, F1Score, MetricCollection
from torch.utils.data import DataLoader
from typing import Optional
import logging
from datetime import datetime
from tqdm import tqdm
import os


class Memory:
    def __init__(self):
        self.best_epoch: int = 0
        self.best_loss: float = float('inf')
        self.patience: Optional[int] = config['model']['early-stopping']

    def update(self, epoch: int, loss: float) -> bool:
        improvement: bool = False
        best_loss = min(self.best_loss, loss)
        if best_loss != self.best_loss:
            improvement = True
            self.best_epoch = epoch
            self.best_loss = loss
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
        for metric_name, score in super(Metrics, self).compute().items():
            logging.info(f'\t->\t{metric_name}: {score.item()}')


class Trainer:
    def __init__(self, model: BaseModel, metrics: Optional[Metrics] = None, save_path: Optional[str] = None):
        logging.info(f"Model '{model.model_name}' has been initialized.")
        self.model: BaseModel = model.to(config['general']['device'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['model']['learning-rate'])
        self.chunk_loss = DiceLoss(ignore_index=int(ChunkCode.NOT_RELEVANT), alpha=0.8)
        self.memory = Memory()
        if metrics is None:
            self.metrics = Metrics(metrics=[
                Precision(num_classes=1),
                Recall(num_classes=1),
                Accuracy(num_classes=1),
                FBetaScore(num_classes=1, beta=0.5),
                F1Score(num_classes=1)
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
            loss = self.chunk_loss(model_out.view([-1, model_out.shape[2]]), chunk_label.view([-1]),
                                   mask=mask.view([-1]))
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            epoch_loss += loss
            bar.set_description(f'Loss: {epoch_loss / (batch_idx + 1):.3f}  ')
        return epoch_loss / len(train_data)

    def _eval(self, epoch: int, dev_data: DataLoader) -> bool:
        if dev_data is not None:
            test_loss: float = self.test(dev_data)
            improvement: bool = self.memory.update(epoch, test_loss)
            if improvement:
                logging.info(f'Improvement has occurred. Saving the model in the path: {self.save_path}')
                self.save_model(self.save_path)
            if self.memory.early_stopping(epoch):
                return True
        return False

    def test(self, test_data: DataLoader) -> float:
        self.model.eval()
        logging.info(f'Test started...')
        test_loss: float = 0.
        with torch.no_grad():
            for batch in tqdm(test_data):
                sentence, chunk_label, mask = batch
                model_out: torch.Tensor = self.model(sentence, mask)
                chunk_label = chunk_label.view([-1])
                test_loss += self.chunk_loss(model_out.view([-1, model_out.shape[-1]]), chunk_label, mask=mask.view([-1]))
                self.metrics(model_out.view([-1]), chunk_label)
            logging.info(f'Test loss: {test_loss / len(test_data)}')
            self.metrics.compute()
            self.metrics.reset()
        return test_loss / len(test_data)

    def save_model(self, save_path: str) -> None:
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, save_path: str) -> None:
        self.model.load_state_dict(torch.load(save_path))

    def predict(self, sentence, mask) -> torch.Tensor:
        self.model.eval()
        return self.model(sentence, mask)
