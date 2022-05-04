import torch
from typing import List

from ASTE.aste.models.model_elements.embeddings import Bert
from ASTE.aste.models.model_elements.span_aggregators import BaseAggregator, EndPointAggregator
from ASTE.aste.models import ModelOutput, ModelLoss, ModelMetric
from ASTE.dataset.reader import Batch


class BaseModel(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

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


from .specialty_models import ChunkerModel, TripletExtractorModel


class BertBaseModel(BaseModel):
    def __init__(self, model_name='Bert Base Model', aggregator: BaseAggregator = EndPointAggregator()):
        super(BertBaseModel, self).__init__(model_name)
        self.embeddings_layer: Bert = Bert()
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
