from ASTE.utils import config
from ASTE.dataset.reader import Batch
from ASTE.dataset.domain.const import ASTELabels
from .crf_model import CRF
from ASTE.aste.tools.metrics import Metric, get_selected_metrics
from ASTE.aste.models import ModelOutput, ModelLoss, ModelMetric, BaseModel
from ASTE.aste.losses import DiceLoss

import torch
from torch.nn import CrossEntropyLoss
from functools import lru_cache
from typing import Tuple, List, Dict


class TripletExtractorModel(BaseModel):
    def __init__(self, input_dim: int, model_name: str = 'Triplet Extractor Model'):
        super(TripletExtractorModel, self).__init__(model_name=model_name)

        self.triplet_loss = DiceLoss(ignore_index=ASTELabels.NOT_RELEVANT,
                                     alpha=config['model']['triplet-extractor']['dice-loss-alpha'])

        metrics: List = get_selected_metrics(num_classes=6, multiclass=True)
        self.independent_metrics: Metric = Metric(name='Independent matrix predictions', metrics=metrics,
                                                  ignore_index=ASTELabels.NOT_RELEVANT).to(config['general']['device'])

        metrics = get_selected_metrics(for_triplets=True)
        self.final_metrics: Metric = Metric(name='Final predictions', metrics=metrics).to(config['general']['device'])

        input_dimension: int = input_dim * 2
        self.linear_layer_1 = torch.nn.Linear(input_dimension, 300)
        self.linear_layer_2 = torch.nn.Linear(300, 100)
        self.linear_layer_3 = torch.nn.Linear(100, 100)
        # 3 is responsible for aspect, opinion, and not_pair
        self.final_layer = torch.nn.Linear(100, config['dataset']['number-of-polarities'] + 3)
        self.crf = CRF()
        self.dropout = torch.nn.Dropout(0.1)
        self.batch_norm = torch.nn.BatchNorm2d(input_dimension)
        self.final_batch_norm = torch.nn.BatchNorm2d(100)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        matrix_data = self.construct_matrix(data)
        matrix_data = self.batch_norm(torch.permute(matrix_data, (0, 3, 1, 2)))
        matrix_data = torch.permute(matrix_data, (0, 2, 3, 1))

        layer: torch.nn.Linear
        for layer in [self.linear_layer_1, self.linear_layer_2, self.linear_layer_3]:
            matrix_data = layer(matrix_data)
            matrix_data = torch.relu(matrix_data)
            matrix_data = self.dropout(matrix_data)

        matrix_data = self.final_batch_norm(torch.permute(matrix_data, (0, 3, 1, 2)))
        matrix_data = torch.permute(matrix_data, (0, 2, 3, 1))
        matrix_data = self.final_layer(matrix_data)
        matrix_data = self.crf(matrix_data)
        return self.softmax(matrix_data)

    @staticmethod
    def construct_matrix(data: torch.Tensor) -> torch.Tensor:
        max_len: int = int(data.shape[1])
        data_col: torch.Tensor = data.unsqueeze(1).expand(-1, max_len, -1, -1)
        data_row: torch.Tensor = data.unsqueeze(2).expand(-1, -1, max_len, -1)
        return torch.cat([data_col, data_row], dim=-1)

    def get_loss(self, model_out: ModelOutput) -> ModelLoss:
        true_labels: torch.Tensor = self.construct_matrix_labels(model_out.batch, tuple(model_out.predicted_spans))
        triplet_loss: torch.Tensor = self.triplet_loss(
            model_out.triplet_results.view([-1, model_out.triplet_results.shape[-1]]),
            true_labels.view([-1])
        )
        return ModelLoss(triplet_extractor_loss=triplet_loss)

    def update_metrics(self, model_out: ModelOutput) -> None:
        true_labels: torch.Tensor = self.construct_matrix_labels(model_out.batch, tuple(model_out.predicted_spans))
        true_triplets: torch.Tensor = self.get_triplets_from_matrix(true_labels)
        total_correct_count: int = self.get_total_correct_triplets_count(model_out.batch)
        predicted_labels: torch.Tensor = torch.argmax(model_out.triplet_results, dim=-1)
        predicted_labels = torch.where(true_labels == ASTELabels.NOT_RELEVANT, true_labels, predicted_labels)
        predicted_triplets: torch.Tensor = self.get_triplets_from_matrix(predicted_labels)

        self.independent_metrics(predicted_labels.view([-1]), true_labels.view([-1]))
        self.final_metrics(predicted_triplets, true_triplets, full_target_count=total_correct_count)

    @staticmethod
    def get_total_correct_triplets_count(batch: Batch) -> int:
        sample: Batch
        return sum([len(sample.sentence_obj[0].triplets) for sample in batch])

    @staticmethod
    @lru_cache(maxsize=None)
    def construct_matrix_labels(batch: Batch, predicted_spans: Tuple[torch.Tensor]) -> torch.Tensor:
        labels_matrix: torch.Tensor = TripletExtractorModel.get_unfilled_labels_matrix(predicted_spans)

        sample_idx: int
        sample: Batch
        for sample_idx, sample in enumerate(batch):
            TripletExtractorModel.fill_one_dim_matrix(sample, labels_matrix[sample_idx, ...],
                                                      predicted_spans[sample_idx])

        return labels_matrix

    @staticmethod
    def get_unfilled_labels_matrix(predicted_spans: Tuple[torch.Tensor]) -> torch.Tensor:
        max_span_num = max([spans.shape[0] for spans in predicted_spans])
        size: Tuple = (len(predicted_spans), max_span_num, max_span_num)
        labels_matrix: torch.Tensor = torch.full(size=size, fill_value=ASTELabels.NOT_PAIR).to(
            config['general']['device'])
        return labels_matrix

    @staticmethod
    def fill_one_dim_matrix(sample: Batch, labels_matrix: torch.Tensor, predicted_spans: torch.Tensor) -> None:

        # TODO make it more readable and add docs
        def check_for_second_element_and_fill_if_necessary(sec_pair: str) -> None:
            assert sec_pair in ('aspect', 'opinion'), f'Invalid second pair source: {sec_pair}!'
            # we can do that in this way because we do not interfere with order of spans
            rows: torch.Tensor = (getattr(sample, f'{sec_pair}_spans')[0][triplet_idx] == predicted_spans).all(dim=1)
            if rows.any():
                sec_span_idx = int(torch.where(rows)[0])
                sentiment: str = sample.sentence_obj[0].triplets[triplet_idx].sentiment
                labels_matrix[span_idx, sec_span_idx] = ASTELabels[sentiment]
                labels_matrix[sec_span_idx, span_idx] = ASTELabels[sentiment]

        span_idx: int
        predicted_span: torch.Tensor
        for span_idx, predicted_span in enumerate(predicted_spans):
            aspect_rows: torch.Tensor = (predicted_span == sample.aspect_spans[0]).all(dim=1)
            opinion_rows: torch.Tensor = (predicted_span == sample.opinion_spans[0]).all(dim=1)
            if aspect_rows.any():
                labels_matrix[span_idx, span_idx] = ASTELabels.ASPECT
                triplet_idx: int
                for triplet_idx in torch.where(aspect_rows)[0]:
                    check_for_second_element_and_fill_if_necessary('opinion')

            elif opinion_rows.any():
                labels_matrix[span_idx, span_idx] = ASTELabels.OPINION
                triplet_idx: int
                for triplet_idx in torch.where(opinion_rows)[0]:
                    check_for_second_element_and_fill_if_necessary('aspect')

        labels_matrix = TripletExtractorModel.mask_one_dim_matrix(labels_matrix, predicted_spans.shape[0])

    @staticmethod
    def mask_one_dim_matrix(labels_matrix: torch.Tensor, span_len: int) -> torch.Tensor:
        mask: torch.Tensor = torch.ones_like(labels_matrix).triu().bool()
        labels_matrix[...] = torch.where(mask, labels_matrix[...], ASTELabels.NOT_RELEVANT)
        labels_matrix[:, span_len:] = ASTELabels.NOT_RELEVANT
        labels_matrix[span_len:, :] = ASTELabels.NOT_RELEVANT
        return labels_matrix

    @staticmethod
    @lru_cache(maxsize=None)
    def get_triplets_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
        triplets: List = list()

        sample_idx: int
        sample: torch.Tensor
        for sample_idx, sample in enumerate(matrix):
            triplets += TripletExtractorModel.get_triplets_from_sample(sample, sample_idx)
        return torch.tensor(triplets).to(config['general']['device'])

    @staticmethod
    def get_triplets_from_sample(sample: torch.Tensor, sample_idx: int = 0) -> List:
        triplets: List = list()
        diag: torch.Tensor = torch.diagonal(sample, 0)
        diag_idx: int
        diag_el: int
        for diag_idx, diag_el in enumerate(diag):
            if diag_el not in (ASTELabels.ASPECT, ASTELabels.OPINION):
                continue
            for col_idx in range(0, diag_idx):
                relation: int = int(sample[col_idx, diag_idx])

                if (diag[col_idx] not in (ASTELabels.ASPECT, ASTELabels.OPINION)) or (
                        diag[col_idx] == diag_el) or relation == ASTELabels.NOT_PAIR:
                    continue

                if diag_el == ASTELabels.ASPECT:
                    triplets.append([sample_idx, diag_idx, col_idx, relation])
                else:
                    triplets.append([sample_idx, col_idx, diag_idx, relation])
        return triplets

    def get_metrics(self) -> ModelMetric:
        metrics: Dict = self.independent_metrics.compute()
        metrics.update(self.final_metrics.compute())
        return ModelMetric(triplet_metric=metrics)

    def reset_metrics(self) -> None:
        self.independent_metrics.reset()
        self.final_metrics.reset()
