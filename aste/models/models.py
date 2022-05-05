import logging

import torch
from typing import List, Dict, Union

from ASTE.aste.models.model_elements.embeddings import Bert, BaseEmbedding
from ASTE.aste.models.model_elements.span_aggregators import BaseAggregator, EndPointAggregator
from ASTE.aste.models import ModelOutput, ModelLoss, ModelMetric
from ASTE.dataset.reader import Batch
from ASTE.utils import config


class BaseModel(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.performed_epochs: int = 0
        self.fully_trainable: bool = True

    def forward(self, *args, **kwargs) -> ModelOutput:
        raise NotImplemented

    def get_loss(self, model_out: ModelOutput) -> ModelLoss:
        raise NotImplemented

    def update_metrics(self, model_out: ModelOutput) -> None:
        raise NotImplemented

    def get_metrics_and_reset(self) -> ModelMetric:
        metrics: ModelMetric = self.get_metrics()
        self.reset_metrics()
        return metrics

    def get_metrics(self) -> ModelMetric:
        raise NotImplemented

    def reset_metrics(self) -> None:
        raise NotImplemented

    def update_trainable_parameters(self) -> None:
        self.performed_epochs += 1


from .specialty_models import ChunkerModel, TripletExtractorModel


class BertBaseModel(BaseModel):
    def __init__(self, model_name='Bert Base Model', aggregator: BaseAggregator = EndPointAggregator()):
        super(BertBaseModel, self).__init__(model_name)
        self.each_part_epochs: Dict = {
            'chunker': config['model']['chunker']['single-model-epochs'],
            'triplet': config['model']['triplet-extractor']['single-model-epochs'],
            'full': config['model']['total-epochs'],
        }

        self.embeddings_layer: BaseEmbedding = Bert()
        self.chunker: BaseModel = ChunkerModel()
        self.aggregator: BaseAggregator = aggregator
        self.triplets_extractor: BaseModel = TripletExtractorModel(embeddings_dim=self.aggregator.embeddings_dim)

    def forward(self, batch: Batch) -> ModelOutput:
        embeddings: torch.Tensor = self.embeddings_layer(batch.sentence, batch.mask)

        chunker_output: torch.Tensor = self.chunker(embeddings)
        predicted_spans: List[torch.Tensor] = self.chunker.get_spans_from_batch_prediction(batch, chunker_output)

        aggregated_embeddings: torch.Tensor = self.aggregator.aggregate(embeddings, predicted_spans)
        triplet_results: torch.Tensor = self.triplets_extractor(aggregated_embeddings)

        return ModelOutput(batch=batch,
                           chunker_output=chunker_output,
                           predicted_spans=predicted_spans,
                           triplet_results=triplet_results)

    def get_loss(self, model_out: ModelOutput) -> ModelLoss:
        return ModelLoss.from_instances(chunker_loss=self.chunker.get_loss(model_out),
                                        triplet_loss=self.triplets_extractor.get_loss(model_out))

    def update_metrics(self, model_out: ModelOutput) -> None:
        self.chunker.update_metrics(model_out)
        self.triplets_extractor.update_metrics(model_out)

    def get_metrics(self) -> ModelMetric:
        return ModelMetric.from_instances(chunker_metric=self.chunker.get_metrics(),
                                          triplet_metric=self.triplets_extractor.get_metrics())

    def reset_metrics(self) -> None:
        self.chunker.reset_metrics()

    def update_trainable_parameters(self) -> None:
        self.performed_epochs += 1
        if self.performed_epochs < self.each_part_epochs['chunker']:
            self.fully_trainable = False
            self.freeze_model(self.triplets_extractor)
            self.unfreeze_model(self.chunker)
            self.unfreeze_model(self.embeddings_layer)
        elif self.performed_epochs < (self.each_part_epochs['triplet'] + self.each_part_epochs['chunker']):
            self.fully_trainable = False
            self.freeze_model(self.chunker)
            self.freeze_model(self.embeddings_layer)
            self.unfreeze_model(self.triplets_extractor)
        elif self.performed_epochs >= self.each_part_epochs['full']:
            self.fully_trainable = True
            self.unfreeze_model(self.triplets_extractor)
            self.unfreeze_model(self.chunker)
            self.unfreeze_model(self.embeddings_layer)

    @staticmethod
    def freeze_model(model: Union[BaseModel, BaseEmbedding]) -> None:
        logging.info(f"Model '{model.model_name}' freeze.")
        if not model.fully_trainable:
            return
        for param in model.parameters():
            param.requires_grad = False
        model.fully_trainable = False

    @staticmethod
    def unfreeze_model(model: Union[BaseModel, BaseEmbedding]) -> None:
        logging.info(f"Model '{model.model_name}' unfreeze.")
        if model.fully_trainable:
            return
        for param in model.parameters():
            param.requires_grad = True
        model.fully_trainable = True
