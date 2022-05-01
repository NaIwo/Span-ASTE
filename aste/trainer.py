from ASTE.aste.models import BaseModel, ModelOutput
from ASTE.utils import config
from ASTE.dataset.domain.const import ChunkCode
from .trainer_tools import Metrics, Memory, BaseTracker
from .losses import DiceLoss
from ASTE.dataset.reader import Batch

import torch
from torchmetrics import FBetaScore, Accuracy, Precision, Recall, F1Score
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Tuple
import logging
from datetime import datetime
from tqdm import tqdm
import os


class Trainer:
    def __init__(self, model: BaseModel, metrics: Optional[Metrics] = None, save_path: Optional[str] = None,
                 tracker: BaseTracker = BaseTracker()):
        logging.info(f"Model '{model.model_name}' has been initialized.")
        self.model: BaseModel = model.to(config['general']['device'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['model']['learning-rate'])
        self.chunk_loss_ignore = DiceLoss(ignore_index=int(ChunkCode.NOT_RELEVANT), alpha=0.)
        self.prediction_threshold: float = 0.5
        self.chunk_loss_lambda: float = 0.001

        self.memory = Memory()
        self.tracker: BaseTracker = tracker
        self.tracker.add_config({'Chunk Loss': {'alpha': self.chunk_loss_ignore.alpha,
                                                'ignore_index': self.chunk_loss_ignore.ignore_index}})

        self.metrics: Metrics = self._get_metrics(metrics)
        self.save_path: str = self._get_save_path(save_path)

    @staticmethod
    def _get_metrics(metrics: Optional[Metrics]) -> Metrics:
        if metrics is None:
            return Metrics(metrics=[
                Precision(num_classes=1, multiclass=False),
                Recall(num_classes=1, multiclass=False),
                Accuracy(num_classes=1, multiclass=False),
                FBetaScore(num_classes=1, multiclass=False, beta=0.5),
                F1Score(num_classes=1, multiclass=False)
            ]).to(config['general']['device'])
        else:
            return metrics.to(config['general']['device'])

    def _get_save_path(self, save_path: Optional[str]) -> str:
        if save_path is None:
            return os.path.join(os.getcwd(), 'results', datetime.now().strftime("%Y%m%d-%H%M%S"),
                                f'{self.model.model_name.replace(" ", "_").lower()}.pth')
        else:
            return save_path

    def set_prediction_threshold(self, prediction_threshold) -> None:
        self.prediction_threshold = prediction_threshold
        logging.info(f'Prediction threshold updated to: {self.prediction_threshold}')

    def train(self, train_data: DataLoader, dev_data: Optional[DataLoader] = None) -> None:
        logging.info(f"Tracker '{self.tracker.name}' has been initialized.")
        self.tracker.init()
        self.tracker.watch(self.model)

        os.makedirs(self.save_path[:self.save_path.rfind(os.sep)], exist_ok=False)
        training_start_time: datetime.time = datetime.now()
        logging.info(f'Training start at time: {training_start_time}')
        for epoch in range(config['model']['epochs']):
            epoch_loss = self._training_epoch(train_data)
            self.tracker.log({'Train Loss': epoch_loss})
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

        batch_idx: int
        batch: Batch
        for batch_idx, batch in enumerate(bar := tqdm(train_data)):
            model_out: ModelOutput = self.model(batch)
            loss = self._get_chunk_loss(model_out.chunker_output, batch)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            epoch_loss += loss
            bar.set_description(f'Loss: {epoch_loss / (batch_idx + 1):.3f}  ')
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
        test_loss: float = 0.
        with torch.no_grad():
            batch: Batch
            for batch in tqdm(test_data):
                model_out: ModelOutput = self.model(batch)
                test_loss += self._get_chunk_loss(model_out.chunker_output, batch)
                self.metrics(
                    self._chunker_out_for_metrics(
                        model_out.chunker_output.view([-1, model_out.chunker_output.shape[-1]]), batch),
                    batch.chunk_label.view([-1]))
            logging.info(f'Test loss: {test_loss / len(test_data):.3f}')
            metric_results: Dict = self.metrics.compute()
            return_values.update(metric_results)
            self.metrics.reset()
        return_values['Test loss'] = test_loss / len(test_data)
        return return_values

    def _get_chunk_loss(self, model_out: torch.Tensor, batch) -> torch.Tensor:
        loss_ignore = self.chunk_loss_ignore(model_out.view([-1, model_out.shape[-1]]), batch.chunk_label.view([-1]))
        chunk_label: torch.Tensor = torch.where(batch.chunk_label != int(ChunkCode.NOT_RELEVANT), batch.chunk_label, 0)
        true_label_sum: torch.Tensor = torch.sum(chunk_label, dim=-1)
        pred_label_sum: torch.Tensor = torch.sum(model_out[..., 1], dim=-1)
        diff: torch.Tensor = torch.abs(pred_label_sum - true_label_sum).type(torch.float)
        return loss_ignore + self.chunk_loss_lambda * torch.mean(diff)

    @staticmethod
    def _chunker_out_for_metrics(chunker_out: torch.Tensor, batch) -> torch.Tensor:
        # Prediction help - if a token consists of several sub-tokens, we certainly do not split in those sub-tokens.
        fill_value: torch.Tensor = torch.zeros(chunker_out.shape[-1]).to(config['general']['device'])
        fill_value[int(ChunkCode.NOT_SPLIT)] = 1.
        sub_mask: torch.Tensor = batch.sub_words_mask.bool().view([-1, 1])
        return torch.where(sub_mask, chunker_out, fill_value)

    def check_coverage_detected_spans(self, data: DataLoader) -> Dict:
        num_predicted: int = 0
        num_correct_predicted: int = 0
        true_num: int = 0
        for batch in tqdm(data):
            for sample in batch:
                model_output: ModelOutput = self.predict(sample)
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

    def predict(self, sample: Batch) -> ModelOutput:
        self.model.eval()
        with torch.no_grad():
            out: ModelOutput = self.model(sample)
        return out
