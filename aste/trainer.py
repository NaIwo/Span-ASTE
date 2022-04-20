import numpy as np

from ASTE.aste.models.base_model import BaseModel
from ASTE.utils import config
from ASTE.dataset.domain.const import ChunkCode
from .trainer_tools import Metrics, Memory, BaseTracker
from .losses import DiceLoss

import torch
from torchmetrics import FBetaScore, Accuracy, Precision, Recall, F1Score
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Tuple, Set
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
        self.chunk_loss_ignore = DiceLoss(ignore_index=int(ChunkCode.NOT_RELEVANT), alpha=0.7)
        self.prediction_threshold: float = 0.5

        self.memory = Memory()
        self.tracker: BaseTracker = tracker
        self.tracker.add_config({'Chunk Loss': {'alpha': self.chunk_loss_ignore.alpha,
                                                'ignore_index': self.chunk_loss_ignore.ignore_index}})

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
        for batch_idx, batch in enumerate(bar := tqdm(train_data)):
            model_out: torch.Tensor = self.model(batch.sentence, batch.mask)
            loss = self._get_chunk_loss(model_out, batch)
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
            for batch in tqdm(test_data):
                model_out: torch.Tensor = self.model(batch.sentence, batch.mask)
                test_loss += self._get_chunk_loss(model_out, batch)
                self.metrics(self._model_out_for_metrics(model_out.view([-1, model_out.shape[-1]]), batch),
                             batch.chunk_label.view([-1]))
            logging.info(f'Test loss: {test_loss / len(test_data):.3f}')
            metric_results: Dict = self.metrics.compute()
            return_values.update(metric_results)
            self.metrics.reset()
        return_values['Test loss'] = test_loss / len(test_data)
        return return_values

    def _get_chunk_loss(self, model_out: torch.Tensor, batch) -> torch.Tensor:
        loss_ignore = self.chunk_loss_ignore(model_out.view([-1, model_out.shape[-1]]), batch.chunk_label.view([-1]))
        # chunk_label: torch.Tensor = torch.where(batch.chunk_label != int(ChunkCode.NOT_RELEVANT), batch.chunk_label, 0)
        # true_label_sum: torch.Tensor = torch.sum(chunk_label, dim=-1)
        # model_out = torch.argmax(model_out, dim=-1)
        # pred_label_sum: torch.Tensor = torch.sum(model_out, dim=-1)
        # diff: torch.Tensor = torch.abs(pred_label_sum - true_label_sum)
        # loss: torch.Tensor = torch.where(diff > 0, diff, torch.pow(diff, 2))
        # loss: torch.Tensor = torch.pow(diff, 2)
        # normalizer: torch.Tensor = torch.sum(batch.mask - 2*batch.sentence_obj[0].encoder.offset)
        return loss_ignore  # + 0.01*torch.pow(torch.sum(diff) / normalizer, 2)

    @staticmethod
    def _model_out_for_metrics(model_out: torch.Tensor, batch) -> torch.Tensor:
        # Prediction help - if a token consists of several sub-tokens, we certainly do not split in those sub-tokens.
        fill_value: torch.Tensor = torch.zeros(model_out.shape[-1]).to(config['general']['device'])
        fill_value[int(ChunkCode.NOT_SPLIT)] = 1.
        sub_mask: torch.Tensor = batch.sub_words_mask.bool().view([-1, 1])
        return torch.where(sub_mask, model_out, fill_value)

    def check_coverage_detected_spans(self, data: DataLoader) -> Dict:
        num_predicted: int = 0
        num_correct_predicted: int = 0
        true_num: int = 0
        for batch in tqdm(data):
            for sample in batch:
                predicted_spans = self._get_predicted_spans(sample)
                true_spans: List[Tuple[int, int]] = sample.sentence_obj[0].get_all_unordered_spans()
                num_correct_predicted += self._count_intersection(true_spans, predicted_spans)
                num_predicted += predicted_spans.shape[0]
                true_num += len(set(true_spans))
        ratio: float = num_correct_predicted / true_num
        logging.info(
            f'Coverage of isolated spans: {ratio}. Extracted spans: {num_predicted}. Total correct spans: {true_num}')
        return {
            'Ratio': ratio,
            'Extracted spans': num_predicted,
            'Total correct spans': true_num
        }

    def _get_predicted_spans(self, sample) -> np.ndarray:
        offset: int = sample.sentence_obj[0].encoder.offset
        predictions: np.ndarray = self.predict(sample.sentence, sample.mask).cpu().numpy()[0]
        predictions = np.where(predictions[:, 1] >= self.prediction_threshold, 1, 0)
        predictions = predictions[:sample.sentence_obj[0].encoded_sentence_length]
        sub_mask: torch.Tensor = sample.sub_words_mask.cpu().numpy()[0][
                                 :sample.sentence_obj[0].encoded_sentence_length]
        # Prediction help - if a token consists of several sub-tokens, we certainly do not split in those sub-tokens.
        predictions = np.where(sub_mask, predictions, int(ChunkCode.NOT_SPLIT))
        # Start and end of spans are the same as start and end of sentence
        predictions = np.pad(np.where(predictions)[0], 1, constant_values=(offset, len(predictions) - offset))
        predicted_spans: np.ndarray = np.lib.stride_tricks.sliding_window_view(predictions, 2)
        # lib.stride_tricks return view of array and we can not manage them as normal array with new shape.
        predicted_spans = np.array(predicted_spans)
        # Because we perform split ->before<- selected word.
        predicted_spans[:, 1] -= 1
        # This deletion is due to offset in padding. Some spans can started from this offset and
        # we could end up with wrong extracted span.
        return np.delete(predicted_spans, np.where(predicted_spans[:, 0] > predicted_spans[:, 1]), axis=0)

    @staticmethod
    def _count_intersection(true_spans: List[Tuple[int, int]], predicted_spans: np.ndarray) -> int:
        predicted_spans: Set = set([(row[0], row[1]) for row in predicted_spans])
        true_spans: Set = set(true_spans)
        return len(predicted_spans.intersection(true_spans))

    def save_model(self, save_path: str) -> None:
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, save_path: str) -> None:
        self.model.load_state_dict(torch.load(save_path))

    def predict(self, sentence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            out: torch.Tensor = self.model(sentence, mask)
        return out
