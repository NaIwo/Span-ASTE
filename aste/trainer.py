from ASTE.utils import config
from .tools import Metrics, Memory, BaseTracker
from ASTE.dataset.reader import Batch
from ASTE.aste.models import ModelOutput, ModelLoss, ModelMetric, BaseModel

import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Tuple
import logging
from datetime import datetime
from tqdm import tqdm
import os


class Trainer:
    def __init__(self, model: BaseModel, save_path: Optional[str] = None,
                 tracker: BaseTracker = BaseTracker()):
        logging.info(f"Model '{model.model_name}' has been initialized.")
        self.model: BaseModel = model.to(config['general']['device'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['model']['learning-rate'])

        self.memory = Memory()
        self.tracker: BaseTracker = tracker

        self.save_path: str = self._get_save_path(save_path)

        self._add_tracker_configs()

    def _get_save_path(self, save_path: Optional[str]) -> str:
        if save_path is None:
            return os.path.join(os.getcwd(), 'results', datetime.now().strftime("%Y%m%d-%H%M%S"),
                                f'{self.model.model_name.replace(" ", "_").lower()}.pth')
        else:
            return save_path

    def _add_tracker_configs(self) -> None:
        self.tracker.add_config({'Chunk Loss': {'alpha': config['model']['chunker']['dice-loss-alpha'],
                                                'lambda factor': config['model']['chunker']['lambda-factor']}})

    def train(self, train_data: DataLoader, dev_data: Optional[DataLoader] = None) -> None:
        logging.info(f"Tracker '{self.tracker.name}' has been initialized.")
        self.tracker.init()
        self.tracker.watch(self.model)

        os.makedirs(self.save_path[:self.save_path.rfind(os.sep)], exist_ok=False)
        training_start_time: datetime.time = datetime.now()
        logging.info(f'Training start at time: {training_start_time}')
        for epoch in range(config['model']['epochs']):
            epoch_loss: ModelLoss = self._training_epoch(train_data)
            self.tracker.log({'Train Loss': epoch_loss})
            logging.info(f"Epoch: {epoch + 1}/{config['model']['epochs']}. Epoch loss: {epoch_loss}")
            early_stopping: bool = self._eval(epoch=epoch, dev_data=dev_data)
            if early_stopping:
                logging.info(f'Early stopping performed. Patience factor: {self.memory.patience}')
                break

        training_stop_time: datetime.time = datetime.now()
        logging.info(f'Training stop at time: {training_stop_time}')
        logging.info(f'Training time in seconds: {(training_stop_time - training_start_time).seconds}')
        logging.info(f'Best epoch: {self.memory.best_epoch}')

    def _training_epoch(self, train_data: DataLoader) -> ModelLoss:
        self.model.train()
        epoch_loss = ModelLoss()

        batch_idx: int
        batch: Batch
        for batch_idx, batch in enumerate(bar := tqdm(train_data)):
            model_out: ModelOutput
            loss: ModelLoss
            model_out, loss = self.model(batch)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            epoch_loss += loss
            bar.set_description(f'Loss: {epoch_loss / (batch_idx + 1)} ')
        return epoch_loss / len(train_data)

    def _eval(self, epoch: int, dev_data: DataLoader) -> bool:
        if dev_data is not None:
            values_dict: Dict = self.test(dev_data)
            self.tracker.log(values_dict)
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
        test_loss = ModelLoss()
        with torch.no_grad():
            batch: Batch
            for batch in tqdm(test_data):
                model_out: ModelOutput
                loss: ModelLoss
                model_out, loss = self.model(batch, compute_metrics=True)
                test_loss += loss
            logging.info(f'Test loss: {test_loss / len(test_data)}')
            metrics: ModelMetric = self.model.get_metrics_and_reset()
            return_values.update(metrics.chunker_metric)
        return_values['Test loss'] = test_loss.full_loss / len(test_data)
        return return_values

    def check_coverage_detected_spans(self, data: DataLoader) -> Dict:
        num_predicted: int = 0
        num_correct_predicted: int = 0
        true_num: int = 0
        for batch in tqdm(data):
            for sample in batch:
                model_output: ModelOutput
                model_output, _, _ = self.predict(sample)
                true_spans: List[Tuple[int, int]] = sample.sentence_obj[0].get_all_unordered_spans()
                true_spans: torch.Tensor = torch.Tensor(true_spans).to(config['general']['device'])
                num_correct_predicted += self._count_intersection(true_spans, model_output.predicted_spans[0])
                num_predicted += model_output.predicted_spans[0].shape[0]
                true_num += len(set(true_spans))
        ratio: float = num_correct_predicted / true_num
        logging.info(
            f'Coverage of isolated spans: {ratio}. Extracted spans: {num_predicted}. Total correct spans: {true_num}')
        return {
            'Ratio': ratio,
            'Extracted spans': num_predicted,
            'Total correct spans': true_num
        }

    @staticmethod
    def _count_intersection(true_spans: torch.Tensor, predicted_spans: torch.Tensor) -> int:
        combined = torch.cat((true_spans, predicted_spans))
        uniques, counts = combined.unique(return_counts=True, dim=0)
        return uniques[counts > 1].shape[0]

    def save_model(self, save_path: str) -> None:
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, save_path: str) -> None:
        self.model.load_state_dict(torch.load(save_path))

    def predict(self, sample: Batch) -> Tuple[ModelOutput, ModelLoss, ModelMetric]:
        self.model.eval()
        with torch.no_grad():
            out: ModelOutput
            loss: ModelLoss
            metrics: ModelMetric
            out, loss = self.model(sample, compute_metrics=True)
            metrics: ModelMetric = self.model.get_metrics_and_reset()
        return out, loss, metrics
