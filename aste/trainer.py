from ASTE.utils import config
from .tools import Memory, BaseTracker
from ASTE.dataset.reader import Batch
from ASTE.aste.models import ModelOutput, ModelLoss, ModelMetric, BaseModel

import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Union
import logging
from datetime import datetime
from tqdm import tqdm
import os
import yaml


class Trainer:
    def __init__(self, model: BaseModel, save_path: Optional[str] = None,
                 tracker: BaseTracker = BaseTracker()):
        logging.info(f"Model '{model.model_name}' has been initialized.")
        self.model: BaseModel = model.to(config['general']['device'])
        self.optimizer = torch.optim.Adam(self.model.get_params_and_lr(), lr=config['model']['learning-rate'])

        self.memory = Memory()
        self.tracker: BaseTracker = tracker

        self.save_path: str = self._get_save_path(save_path)

    def _get_save_path(self, save_path: Optional[str]) -> str:
        if save_path is None:
            return os.path.join(os.getcwd(), 'results', datetime.now().strftime("%Y%m%d-%H%M%S"),
                                f'{self.model.model_name.replace(" ", "_").lower()}.pth')
        else:
            return save_path

    def train(self, train_data: DataLoader, dev_data: Optional[DataLoader] = None) -> None:
        logging.info(f"Tracker '{self.tracker.name}' has been initialized.")
        self.tracker.init()
        self.tracker.watch(self.model)

        os.makedirs(self.save_path[:self.save_path.rfind(os.sep)], exist_ok=False)
        training_start_time: datetime.time = datetime.now()
        logging.info(f'Training start at time: {training_start_time}')
        for epoch in range(config['model']['total-epochs']):
            self.model.update_trainable_parameters()
            if self.model.warmup:
                self.memory.reset()
            epoch_loss: ModelLoss = self._training_epoch(train_data)
            self.tracker.log({'Train Loss': epoch_loss.logs})
            logging.info(f"Epoch: {epoch + 1}/{config['model']['total-epochs']}. Epoch loss: {epoch_loss}")
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
            model_out: ModelOutput = self.model(batch)
            loss = self.model.get_loss(model_out)
            loss.backward()
            if ((batch_idx + 1) % config['dataset']['effective-batch-size'] == 0) or (batch_idx + 1 == len(train_data)):
                self.optimizer.step()
                self.optimizer.zero_grad()
            epoch_loss += loss.items()
            bar.set_description(f'Loss: {epoch_loss / (batch_idx + 1)} ')
        return epoch_loss / len(train_data)

    def _eval(self, epoch: int, dev_data: DataLoader) -> bool:
        if dev_data is not None:
            values_dict: Dict = dict()
            eval_out: Dict[str, Union[ModelOutput, ModelLoss, ModelMetric]] = self.test(dev_data)
            values_dict.update(eval_out[ModelMetric.NAME].triplet_metric)
            values_dict['Test loss'] = eval_out[ModelLoss.NAME].full_loss
            self.tracker.log({'Test Loss': eval_out[ModelLoss.NAME].logs,
                              'Metrics': eval_out[ModelMetric.NAME].logs})
            improvement: bool = self.memory.update(epoch, values_dict)
            if improvement:
                logging.info(f'Improvement has occurred. Saving the model in the path: {self.save_path}')
                self.save_model(self.save_path)
            if self.memory.early_stopping(epoch):
                return True
        return False

    def test(self, test_data: DataLoader) -> Dict[str, Union[ModelOutput, ModelLoss, ModelMetric]]:
        self.model.eval()
        logging.info(f'Test started...')
        test_loss = ModelLoss()
        with torch.no_grad():
            batch_idx: int
            batch: Batch
            for batch_idx, batch in enumerate(bar := tqdm(test_data)):
                model_out: ModelOutput = self.model(batch)
                self.model.update_metrics(model_out)
                loss: ModelLoss = self.model.get_loss(model_out)
                test_loss += loss.items()
                bar.set_description(f'Test Loss: {test_loss / (batch_idx + 1)}')
            logging.info(f'Test loss: {test_loss / len(test_data)}')
            metrics: ModelMetric = self.model.get_metrics_and_reset()
        self.pprint_metrics(metrics)
        test_loss = test_loss / len(test_data)
        return {
            ModelOutput.NAME: model_out,
            ModelLoss.NAME: test_loss,
            ModelMetric.NAME: metrics
        }

    @staticmethod
    def pprint_metrics(metrics: ModelMetric) -> None:
        logging.info(f'\n{ModelMetric.NAME}\n'
                     f'{yaml.dump(metrics.__dict__, sort_keys=False, default_flow_style=False)}')

    def check_coverage_detected_spans(self, data: DataLoader) -> Dict:
        num_predicted: int = 0
        num_correct_predicted: int = 0
        true_num: int = 0
        for batch in tqdm(data):
            sample: Batch
            for sample in batch:
                model_output: ModelOutput = self.predict(sample)
                true_spans: torch.Tensor = torch.cat([sample.aspect_spans[0], sample.opinion_spans[0]], dim=0).unique(
                    dim=0)
                num_correct_predicted += self._count_intersection(true_spans, model_output.predicted_spans[0])
                num_predicted += model_output.predicted_spans[0].shape[0]
                true_num += true_spans.shape[0] - int(-1 in true_spans)
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
        all_spans: torch.Tensor = torch.cat([true_spans, predicted_spans], dim=0)
        uniques, counts = torch.unique(all_spans, return_counts=True, dim=0)
        return uniques[counts > 1].shape[0]

    def save_model(self, save_path: str) -> None:
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, save_path: str) -> None:
        self.model.load_state_dict(torch.load(save_path))

    def predict(self, sample: Batch) -> ModelOutput:
        self.model.eval()
        with torch.no_grad():
            out: ModelOutput = self.model(sample)
        return out
