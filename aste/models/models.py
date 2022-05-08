import logging

import torch
from typing import List, Dict

from ASTE.aste.models.model_elements.embeddings import Bert, BaseEmbedding
from ASTE.aste.models.model_elements.span_aggregators import BaseAggregator, EndPointAggregator
from ASTE.aste.models import ModelOutput, ModelLoss, ModelMetric
from ASTE.dataset.reader import Batch


class BaseModel(torch.nn.Module):
    def __init__(self, model_name: str, *args, **kwargs):
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

    def freeze(self) -> None:
        logging.info(f"Model '{self.model_name}' freeze.")
        if not self.fully_trainable:
            return
        for param in self.parameters():
            param.requires_grad = False
        self.fully_trainable = False

    def unfreeze(self) -> None:
        logging.info(f"Model '{self.model_name}' unfreeze.")
        if self.fully_trainable:
            return
        for param in self.parameters():
            param.requires_grad = True
        self.fully_trainable = True


from .specialty_models import ChunkerModel, TripletExtractorModel, SpanModel, Selector


class BertBaseModel(BaseModel):
    def __init__(self, model_name='Bert Base Model', *args, **kwargs):

        super(BertBaseModel, self).__init__(model_name)
        self.each_part_epochs: Dict = {
            'chunker and selector': 10,
            'triplet': 5
        }

        self.embeddings_layer_chunker: BaseEmbedding = Bert()
        self.chunker: BaseModel = ChunkerModel(input_dim=self.embeddings_layer_chunker.embedding_dim)
        self.aggregator: BaseAggregator = EndPointAggregator(input_dim=self.embeddings_layer_chunker.embedding_dim)
        self.span_selector: BaseModel = Selector(input_dim=self.aggregator.output_dim)
        self.triplets_extractor: BaseModel = TripletExtractorModel(input_dim=self.aggregator.output_dim)

    def forward(self, batch: Batch) -> ModelOutput:
        embeddings_chunker: torch.Tensor = self.embeddings_layer_chunker(batch.sentence, batch.mask)

        chunker_output: torch.Tensor = self.chunker(embeddings_chunker)
        predicted_spans: List[torch.Tensor] = self.chunker.get_spans_from_batch_prediction(batch, chunker_output)

        agg_emb_chunker: torch.Tensor = self.aggregator.aggregate(embeddings_chunker, predicted_spans)

        span_selector_output: torch.Tensor = self.span_selector(agg_emb_chunker)
        triplet_input: torch.Tensor = span_selector_output[..., 1:] * agg_emb_chunker

        triplet_results: torch.Tensor = self.triplets_extractor(triplet_input)

        return ModelOutput(batch=batch,
                           chunker_output=chunker_output,
                           predicted_spans=predicted_spans,
                           span_selector_output=span_selector_output,
                           triplet_results=triplet_results)

    def get_loss(self, model_out: ModelOutput) -> ModelLoss:
        return ModelLoss.from_instances(chunker_loss=self.chunker.get_loss(model_out) * self.chunker.fully_trainable,
                                        triplet_loss=self.triplets_extractor.get_loss(
                                            model_out) * self.triplets_extractor.fully_trainable,
                                        span_selector_loss=self.span_selector.get_loss(
                                            model_out) * self.span_selector.fully_trainable)

    def update_metrics(self, model_out: ModelOutput) -> None:
        self.chunker.update_metrics(model_out)
        self.triplets_extractor.update_metrics(model_out)
        self.span_selector.update_metrics(model_out)

    def get_metrics(self) -> ModelMetric:
        return ModelMetric.from_instances(chunker_metric=self.chunker.get_metrics(),
                                          triplet_metric=self.triplets_extractor.get_metrics(),
                                          span_selector_metric=self.span_selector.get_metrics())

    def reset_metrics(self) -> None:
        self.chunker.reset_metrics()

    def update_trainable_parameters(self) -> None:
        self.performed_epochs += 1
        if self.performed_epochs < self.each_part_epochs['chunker and selector']:
            self.fully_trainable = False
            self.triplets_extractor.freeze()
            self.span_selector.unfreeze()
            self.chunker.unfreeze()
            self.embeddings_layer_chunker.unfreeze()
        elif self.performed_epochs < (self.each_part_epochs['triplet'] + self.each_part_epochs['chunker and selector']):
            self.fully_trainable = False
            self.chunker.freeze()
            self.span_selector.freeze()
            self.embeddings_layer_chunker.freeze()
            self.triplets_extractor.unfreeze()
        elif self.performed_epochs >= (self.each_part_epochs['triplet'] + self.each_part_epochs['chunker and selector']):
            self.fully_trainable = True
            self.triplets_extractor.unfreeze()
            self.span_selector.unfreeze()
            self.chunker.unfreeze()
            self.embeddings_layer_chunker.unfreeze()
