import torch
from torch import Tensor
from typing import List, Dict, TypeVar, Optional, Tuple
from functools import lru_cache
import json
import os

from ASTE.utils import config
from ASTE.dataset.reader import Batch
from ASTE.dataset.domain.const import ASTELabels

ML = TypeVar('ML', bound='ModelLoss')
MM = TypeVar('MM', bound='ModelMetric')


class ModelOutput:
    NAME: str = 'Outputs'

    def __init__(self, batch: Batch, chunker_output: Tensor, predicted_spans: List[Tensor],
                 span_selector_output: Tensor, triplet_results: Tensor):
        self.batch: Batch = batch
        self.chunker_output: Tensor = chunker_output
        self.predicted_spans: List[Tensor] = predicted_spans
        self.span_selector_output: Tensor = span_selector_output
        self.triplet_results: Tensor = triplet_results

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self):
        return str(self.result)

    def save(self, path: str) -> None:
        os.makedirs(path[:path.rfind(os.sep)], exist_ok=True)
        triplets: List = self._triplet_results_for_save()
        idx: int
        triplet: List
        with open(path, 'a') as f:
            for idx, triplet in enumerate(triplets):
                line: str = f'{self.batch.sentence_obj[idx].sentence}{self.batch.sentence_obj[idx].SEP}{str(triplet)}\n'
                f.write(line)

    @property
    def result(self) -> Dict:
        triplets: List = self._triplet_results_for_save()
        results: Dict = dict()

        idx: int
        triplet: List
        for idx, triplet in enumerate(triplets):
            results[self.batch.sentence_obj[idx].sentence] = triplet

        return results

    @lru_cache(maxsize=None)
    def _triplet_results_for_save(self) -> List:
        triplet_results: List = [[] for _ in range(len(self.batch))]
        triplets: Tensor = self._get_triplets_from_matrix()

        for triplet in triplets:
            sent_idx: int = int(triplet[0].cpu())
            spans: Tensor = self.predicted_spans[sent_idx]

            aspect_span: List = spans[triplet[1]].cpu().numpy().tolist()
            aspect_span = self.get_result_spans(aspect_span, sent_idx)

            opinion_span: List = spans[triplet[2]].cpu().numpy().tolist()
            opinion_span = self.get_result_spans(opinion_span, sent_idx)

            result: Tuple = (aspect_span, opinion_span, ASTELabels(int(triplet[-1].cpu())).name)
            triplet_results[sent_idx].append(result)

        return triplet_results

    def _get_triplets_from_matrix(self) -> Tensor:
        from .specialty_models import TripletExtractorModel as TEM

        predicted_labels: torch.Tensor = torch.argmax(self.triplet_results, dim=-1)

        idx: int
        sample: Tensor
        for idx, sample in enumerate(predicted_labels):
            predicted_labels[idx, ...] = TEM.mask_one_dim_matrix(sample, self.predicted_spans[idx].shape[0])
        triplets: Tensor = TEM.get_triplets_from_matrix(predicted_labels)
        return triplets

    def get_result_spans(self, curr_span: List, sent_idx: int) -> List:
        curr_span = [
            self.batch.sentence_obj[sent_idx].get_index_before_encoding(curr_span[0]),
            self.batch.sentence_obj[sent_idx].get_index_before_encoding(curr_span[1])
        ]
        return [curr_span[0]] if curr_span[0] == curr_span[1] else [curr_span[0], curr_span[1]]


ZERO: Tensor = torch.tensor(0., device=config['general']['device'])


class ModelLoss:
    NAME: str = 'Losses'

    def __init__(self, *, chunker_loss: Tensor = ZERO, span_selector_loss: Tensor = ZERO,
                 triplet_extractor_loss: Tensor = ZERO, weighted: bool = True):
        self.chunker_loss: Tensor = chunker_loss
        self.span_selector_loss: Tensor = span_selector_loss
        self.triplet_extractor_loss: Tensor = triplet_extractor_loss

        if weighted:
            self._include_weights()

    @classmethod
    def from_instances(cls, *, chunker_loss: ML, triplet_extractor_loss: ML, span_selector_loss: ML,
                       weighted: bool = False) -> ML:
        return cls(
            chunker_loss=chunker_loss.chunker_loss,
            span_selector_loss=span_selector_loss.span_selector_loss,
            triplet_extractor_loss=triplet_extractor_loss.triplet_extractor_loss,
            weighted=weighted
        )

    def _include_weights(self) -> None:
        self.chunker_loss *= config['model']['chunker']['loss-weight']
        self.span_selector_loss *= config['model']['selector']['loss-weight']
        self.triplet_extractor_loss *= config['model']['triplet-extractor']['loss-weight']

    def backward(self) -> None:
        self.full_loss.backward()

    def items(self) -> ML:
        self.detach()
        return self

    def detach(self) -> None:
        self.chunker_loss = self.chunker_loss.detach()
        self.span_selector_loss = self.span_selector_loss.detach()
        self.triplet_extractor_loss = self.triplet_extractor_loss.detach()

    @property
    def full_loss(self) -> Tensor:
        return self.chunker_loss + self.span_selector_loss + self.triplet_extractor_loss

    @property
    def _loss_dict(self) -> Dict:
        return {
            'chunker_loss': float(self.chunker_loss),
            'span_selector_loss': float(self.span_selector_loss),
            'triplet_extractor_loss': float(self.triplet_extractor_loss),
            'full_loss': float(self.full_loss)
        }

    def to_json(self, path: str) -> None:
        os.makedirs(path[:path.rfind(os.sep)], exist_ok=True)
        with open(path, 'a') as f:
            json.dump(self._loss_dict, f)

    def __radd__(self, other: ML) -> ML:
        return self.__add__(other)

    def __add__(self, other: ML) -> ML:
        return ModelLoss(
            chunker_loss=self.chunker_loss + other.chunker_loss,
            span_selector_loss=self.span_selector_loss + other.span_selector_loss,
            triplet_extractor_loss=self.triplet_extractor_loss + other.triplet_extractor_loss,
            weighted=False
        )

    def __truediv__(self, other: float) -> ML:
        return ModelLoss(
            chunker_loss=self.chunker_loss / other,
            span_selector_loss=self.span_selector_loss / other,
            triplet_extractor_loss=self.triplet_extractor_loss / other,
            weighted=False
        )

    def __rmul__(self, other: float) -> ML:
        return self.__mul__(other)

    def __mul__(self, other: float) -> ML:
        return ModelLoss(
            chunker_loss=self.chunker_loss * other,
            span_selector_loss=self.span_selector_loss * other,
            triplet_extractor_loss=self.triplet_extractor_loss * other,
            weighted=False
        )

    def __iter__(self):
        for element in self._loss_dict.items():
            yield element

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self):
        return str({name: round(value, 5) for name, value in self._loss_dict.items()})

    @property
    def logs(self) -> Dict:
        return self._loss_dict


class ModelMetric:
    NAME: str = 'Metrics'

    def __init__(self, *, chunker_metric: Optional[Dict] = None, span_selector_metric: Optional[Dict] = None,
                 triplet_metric: Optional[Dict] = None):
        self.chunker_metric: Optional[Dict] = chunker_metric
        self.span_selector_metric: Optional[Dict] = span_selector_metric
        self.triplet_metric: Optional[Dict] = triplet_metric

    @classmethod
    def from_instances(cls, *, chunker_metric: MM, triplet_metric: MM, span_selector_metric: MM) -> MM:
        return cls(
            chunker_metric=chunker_metric.chunker_metric,
            span_selector_metric=span_selector_metric.span_selector_metric,
            triplet_metric=triplet_metric.triplet_metric
        )

    @property
    def _all_metrics(self) -> Dict:
        return {
            'chunker_metrics': self.chunker_metric,
            'span_selector_metric': self.span_selector_metric,
            'triplet_metric': self.triplet_metric
        }

    def to_json(self, path: str) -> None:
        os.makedirs(path[:path.rfind(os.sep)], exist_ok=True)
        with open(path, 'a') as f:
            json.dump(self._all_metrics, f)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self):
        return str(self._all_metrics)

    def __iter__(self):
        for metrics in self._all_metrics:
            yield metrics

    @property
    def logs(self) -> Dict:
        return self._all_metrics
