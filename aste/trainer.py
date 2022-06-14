from ASTE.utils import config
from .tools import Memory, BaseTracker
from ASTE.dataset.reader import Batch
from ASTE.aste.models import ModelOutput, ModelLoss, ModelMetric, BaseModel
from ASTE.dataset.domain.sentence import Sentence

import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Union, Any
from functools import singledispatchmethod
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

        os.makedirs(self.save_path[:self.save_path.rfind(os.sep)], exist_ok=True)
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
        logging.info(f'Best epoch: {self.memory.best_epoch}')
        logging.info(f'Training stop at time: {training_stop_time}')
        logging.info(f'Training time in seconds: {(training_stop_time - training_start_time).seconds}')
        self.load_model(self.save_path)
        logging.info(f'Best model weights loaded.')
        self.tracker.finish()

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
            bar.set_description(f'Loss: {epoch_loss.full_loss / (batch_idx + 1)} ')
        return epoch_loss / len(train_data)

    def _eval(self, epoch: int, dev_data: DataLoader) -> bool:
        if dev_data is not None:
            values_dict: Dict = dict()
            eval_out: Dict[str, Union[ModelLoss, ModelMetric]] = self.test(dev_data)
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

    @torch.no_grad()
    def test(self, test_data: DataLoader) -> Dict[str, Union[ModelLoss, ModelMetric]]:
        self.model.eval()
        logging.info(f'Test started...')
        test_loss = ModelLoss()
        batch_idx: int
        batch: Batch
        for batch_idx, batch in enumerate(bar := tqdm(test_data)):
            model_out: ModelOutput = self.model(batch)
            self.model.update_metrics(model_out)
            loss: ModelLoss = self.model.get_loss(model_out)
            test_loss += loss.items()
            bar.set_description(f'Test Loss: {test_loss.full_loss / (batch_idx + 1)}')
        logging.info(f'Test loss: {test_loss / len(test_data)}')
        metrics: ModelMetric = self.model.get_metrics_and_reset()
        self.pprint_metrics(metrics)
        test_loss = test_loss / len(test_data)
        return {
            ModelLoss.NAME: test_loss,
            ModelMetric.NAME: metrics
        }

    @staticmethod
    def pprint_metrics(metrics: ModelMetric) -> None:
        logging.info(f'\n{ModelMetric.NAME}\n'
                     f'{yaml.dump(metrics.__dict__, sort_keys=False, default_flow_style=False)}')

    def __getattr__(self, func_name) -> Any:
        def wrapper(*args, **kwargs):
            self.model.eval()
            return getattr(self.model, func_name)(*args, **kwargs)

        return wrapper

    def save_model(self, save_path: str) -> None:
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, save_path: str) -> None:
        self.model.load_state_dict(torch.load(save_path), strict=False)

    @singledispatchmethod
    def predict(self, sample: Union[Batch, Sentence]) -> ModelOutput:
        raise NotImplementedError(f'Cannot make a prediction on the passed input data type: {type(sample)}')

    @predict.register
    @torch.no_grad()
    def predict_batch(self, sample: Batch) -> ModelOutput:
        self.model.eval()
        out: ModelOutput = self.model(sample)
        return out

    @predict.register
    @torch.no_grad()
    def predict_sentence(self, sample: Sentence) -> ModelOutput:
        sample = Batch.from_sentence(sample)
        self.model.eval()
        out: ModelOutput = self.model(sample)
        return out
