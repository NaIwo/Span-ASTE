import torch
from typing import Tuple, List

from ASTE.aste.model_elements.embeddings.bert_embeddings import Bert
from ASTE.aste.model_elements.span_aggregators import BaseAggregator, EndPointAggregator
from ASTE.aste.models import ModelOutput, ModelLoss, ModelMetric
from ASTE.dataset.reader import Batch


class BaseModel(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

    def forward(self, *args, **kwargs) -> Tuple[ModelOutput, ModelLoss]:
        raise NotImplemented


from .specialty_models.chunker_model import ChunkerModel


class BertBaseModel(BaseModel):
    def __init__(self, model_name='Bert Base Model'):
        super(BertBaseModel, self).__init__(model_name)
        self.embeddings_layer: Bert = Bert()
        self.chunker: BaseModel = ChunkerModel()
        self.aggregator: BaseAggregator = EndPointAggregator()

    def forward(self, batch: Batch, compute_metrics: bool = False) -> ModelOutput:
        embeddings: torch.Tensor = self.embeddings_layer(batch.sentence, batch.mask)

        chunker_output: torch.Tensor = self.chunker(embeddings)
        predicted_spans: List[torch.Tensor] = self.chunker.get_spans_from_batch_prediction(batch, chunker_output)

        aggregated_embeddings: torch.Tensor = self.aggregator.aggregate(embeddings, predicted_spans)

        if compute_metrics:
            self.chunker.update_metrics(batch, chunker_output)

        return ModelOutput(batch=batch,
                           chunker_output=chunker_output,
                           predicted_spans=predicted_spans)

    def get_loss(self, model_out: ModelOutput) -> ModelLoss:
        return ModelLoss(chunker_loss=self.chunker.loss(model_out.batch, model_out.chunker_output))

    def get_metrics_and_reset(self) -> ModelMetric:
        metrics: ModelMetric = self.get_metrics()
        self.reset_metrics()
        return metrics

    def get_metrics(self) -> ModelMetric:
        return ModelMetric(chunker_metric=self.chunker.metrics.compute())

    def reset_metrics(self) -> None:
        self.chunker.metrics.reset()
