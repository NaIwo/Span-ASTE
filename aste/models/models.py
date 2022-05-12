import torch
from typing import List, Dict

from ASTE.aste.models.model_elements.embeddings import Bert, BaseEmbedding
from ASTE.aste.models.model_elements.span_aggregators import BaseAggregator, EndPointAggregator
from ASTE.aste.models import ModelOutput, ModelLoss, ModelMetric, BaseModel
from ASTE.dataset.reader import Batch
from ASTE.utils import config
from .specialty_models import ChunkerModel, TripletExtractorModel, Selector, CRF


class BertBaseModel(BaseModel):
    def __init__(self, model_name='Bert Base Model', *args, **kwargs):

        super(BertBaseModel, self).__init__(model_name)
        self.emb_layer: BaseEmbedding = Bert()
        self.chunker: BaseModel = ChunkerModel(input_dim=self.emb_layer.embedding_dim)
        self.aggregator: BaseAggregator = EndPointAggregator(input_dim=self.emb_layer.embedding_dim)
        self.span_selector: BaseModel = Selector(input_dim=self.aggregator.output_dim)
        self.triplets_extractor: BaseModel = TripletExtractorModel(input_dim=self.aggregator.output_dim)
        self.crf: BaseModel = CRF()

        epochs: List = [12, 15, 35, config['model']['total-epochs']]
        self.training_scheduler: Dict = {
            range(0, epochs[0]): {
                'freeze': [self.triplets_extractor, self.crf],
                'unfreeze': [self.chunker, self.span_selector, self.emb_layer],
                'warmup': [self, self.chunker, self.span_selector, self.triplets_extractor, self.crf],
                'not_warmup': []
            },
            range(epochs[0], epochs[1]): {
                'freeze': [self.chunker, self.span_selector, self.emb_layer],
                'unfreeze': [self.triplets_extractor, self.crf],
                'warmup': [self, self.chunker, self.span_selector, self.triplets_extractor, self.crf],
                'not_warmup': []
            },
            range(epochs[1], epochs[2]): {
                'freeze': [],
                'unfreeze': [self.chunker, self.span_selector, self.triplets_extractor, self.emb_layer, self.crf],
                'warmup': [self.triplets_extractor],
                'not_warmup': [self, self.chunker, self.span_selector, self.crf]
            },
            range(epochs[2], epochs[3]): {
                'freeze': [],
                'unfreeze': [self.chunker, self.span_selector, self.triplets_extractor, self.emb_layer, self.crf],
                'warmup': [],
                'not_warmup': [self, self.chunker, self.span_selector, self.triplets_extractor, self.crf]
            }
        }

    def forward(self, batch: Batch) -> ModelOutput:
        embeddings_chunker: torch.Tensor = self.emb_layer(batch.sentence, batch.mask)

        chunker_output: torch.Tensor = self.chunker(embeddings_chunker)
        predicted_spans: List[torch.Tensor] = self.chunker.get_spans_from_batch_prediction(batch, chunker_output)

        agg_emb_chunker: torch.Tensor = self.aggregator.aggregate(embeddings_chunker, predicted_spans)

        span_selector_output: torch.Tensor = self.span_selector(agg_emb_chunker)
        triplet_input: torch.Tensor = span_selector_output[..., 1:] * agg_emb_chunker

        triplet_results: torch.Tensor = self.triplets_extractor(triplet_input)
        crf_results: torch.Tensor = self.crf(triplet_results)

        return ModelOutput(batch=batch,
                           chunker_output=chunker_output,
                           predicted_spans=predicted_spans,
                           span_selector_output=span_selector_output,
                           triplet_results=triplet_results,
                           crf_results=crf_results)

    def get_loss(self, model_out: ModelOutput) -> ModelLoss:
        return ModelLoss.from_instances(chunker_loss=self.chunker.get_loss(model_out) * self.chunker.trainable,
                                        triplet_loss=self.triplets_extractor.get_loss(
                                            model_out) * self.triplets_extractor.trainable,
                                        span_selector_loss=self.span_selector.get_loss(
                                            model_out) * self.span_selector.trainable,
                                        crf_loss=self.crf.get_loss(model_out) * self.crf.trainable)

    def update_metrics(self, model_out: ModelOutput) -> None:
        self.chunker.update_metrics(model_out)
        self.triplets_extractor.update_metrics(model_out)
        self.span_selector.update_metrics(model_out)
        self.crf.update_metrics(model_out)

    def get_metrics(self) -> ModelMetric:
        return ModelMetric.from_instances(chunker_metric=self.chunker.get_metrics(),
                                          triplet_metric=self.triplets_extractor.get_metrics(),
                                          span_selector_metric=self.span_selector.get_metrics(),
                                          crf_metric=self.crf.get_metrics())

    def reset_metrics(self) -> None:
        self.chunker.reset_metrics()

    def get_params_and_lr(self) -> List[Dict]:
        return [
            {'params': self.chunker.parameters(), 'lr': config['model']['learning-rate']},
            {'params': self.aggregator.parameters(), 'lr': config['model']['learning-rate']},
            {'params': self.span_selector.parameters(), 'lr': config['model']['learning-rate']},
            {'params': self.triplets_extractor.parameters(), 'lr': config['model']['learning-rate']},
            {'params': self.crf.parameters(), 'lr': config['model']['learning-rate']},
            {'params': self.emb_layer.parameters(), 'lr': config['model']['bert']['learning-rate']}
        ]

    def update_trainable_parameters(self) -> None:
        model: BaseModel
        for keys, values in self.training_scheduler.items():
            if self.performed_epochs in keys:
                [model.unfreeze() for model in values['unfreeze']]
                [model.freeze() for model in values['freeze']]
                [setattr(model, 'warmup', True) for model in values['warmup']]
                [setattr(model, 'warmup', False) for model in values['not_warmup']]

        self.performed_epochs += 1
