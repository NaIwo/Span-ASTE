from ASTE.utils import config
from ASTE.dataset.reader import Batch
from ASTE.dataset.domain.const import ChunkCode
from ASTE.dataset.domain.sentence import Sentence
from ASTE.aste.losses import DiceLoss
from ASTE.aste.tools.metrics import Metric, get_selected_metrics
from ASTE.aste.models import ModelOutput, ModelLoss, ModelMetric, BaseModel

import torch
from typing import List


class ChunkerModel(BaseModel):
    def __init__(self, input_dim: int, model_name: str = 'Chunker Model'):
        super(ChunkerModel, self).__init__(model_name)
        self.chunk_loss_ignore = DiceLoss(ignore_index=ChunkCode.NOT_RELEVANT,
                                          alpha=config['model']['chunker']['dice-loss-alpha'])
        self.loss_lambda: float = config['model']['chunker']['lambda-factor']
        self.metrics: Metric = Metric(name='Chunker', metrics=get_selected_metrics(for_spans=True)).to(
            config['general']['device'])

        self.mode: int = self._get_mode()

        self.dropout = torch.nn.Dropout(0.1)
        self.linear_layer_1 = torch.nn.Linear(input_dim, 400)
        self.linear_layer_2 = torch.nn.Linear(400, 100)
        self.linear_layer_3 = torch.nn.Linear(100, 10)
        self.final_layer = torch.nn.Linear(10, 2)
        self.softmax = torch.nn.Softmax(dim=-1)

    @staticmethod
    def _get_mode() -> int:
        assert config['model']['chunker']['mode'].lower() in (
            'soft', 'hard'), f"Incorrect chunker operation mode: {config['model']['chunker']['mode']}"
        return 1 if config['model']['chunker']['mode'].lower() == 'soft' else 0

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        layer: torch.nn.Linear
        for layer in [self.linear_layer_1, self.linear_layer_2, self.linear_layer_3]:
            data = layer(data)
            data = torch.relu(data)
            data = self.dropout(data)
        data = self.final_layer(data)
        return self.softmax(data)

    @staticmethod
    def get_spans_from_batch_prediction(batch: Batch, predictions: torch.Tensor) -> List[torch.Tensor]:
        results: List[torch.Tensor] = list()
        predictions: torch.Tensor = torch.argmax(predictions, dim=-1)

        sample: Batch
        prediction: torch.Tensor
        for sample, prediction in zip(batch, predictions):
            predicted_spans: torch.Tensor = ChunkerModel._get_spans_from_single_sample(sample, prediction)
            results.append(predicted_spans)
        return results

    @staticmethod
    def _get_spans_from_single_sample(sample: Batch, prediction: torch.Tensor) -> torch.Tensor:
        prediction = ChunkerModel._get_chunk_indexes(sample, prediction)
        predicted_spans = torch.combinations(prediction)
        # Because we perform split ->before<- selected word.
        predicted_spans[:, 1] -= 1
        max_len: int = config['model']['chunker']['max-len']
        lengths: torch.Tensor = predicted_spans[:, 1] - predicted_spans[:, 0]
        proper_indexes: torch.Tensor = torch.where(lengths <= max_len)[0]
        predicted_spans = predicted_spans[proper_indexes]
        predicted_spans = ChunkerModel._handle_empty_tensor(sample, predicted_spans)

        return predicted_spans.to(torch.int32)

    @staticmethod
    def _get_chunk_indexes(sample: Batch, prediction: torch.Tensor) -> torch.Tensor:
        offset: int = sample.sentence_obj[0].encoder.offset
        prediction = prediction[:sample.sentence_obj[0].encoded_sentence_length]
        # Prediction help - if a token consists of several sub-tokens, we certainly do not split in those sub-tokens
        max_token: int = len(prediction)
        indexes: torch.Tensor = torch.where(prediction)[0]
        s: Sentence = sample.sentence_obj[0]
        prediction: torch.Tensor = torch.tensor(
            [s.get_index_after_encoding(s.get_index_before_encoding(idx)) for idx in indexes],
            device=config['general']['device'])

        # Start and end of spans are the same as start and end of sentence
        prediction = torch.nn.functional.pad(prediction, [1, 0], mode='constant', value=offset)
        prediction = torch.nn.functional.pad(prediction, [0, 1], mode='constant', value=max_token - offset)
        return torch.unique(prediction)

    @staticmethod
    def _handle_empty_tensor(sample: Batch, predicted_spans: torch.Tensor) -> torch.Tensor:
        if not predicted_spans.nelement():
            offset: int = sample.sentence_obj[0].encoder.offset
            predicted_spans = torch.tensor([[offset, sample.sentence_obj[0].encoded_sentence_length - offset - 1]],
                                           device=config['general']['device'])
        return predicted_spans

    def get_loss(self, model_out: ModelOutput) -> ModelLoss:
        predictions: torch.Tensor = model_out.chunker_output
        loss_ignore = self.chunk_loss_ignore(predictions.view([-1, predictions.shape[-1]]),
                                             model_out.batch.chunk_label.view([-1]))
        chunk_label: torch.Tensor = torch.where(model_out.batch.chunk_label != ChunkCode.NOT_RELEVANT,
                                                model_out.batch.chunk_label, 0)
        true_label_sum: torch.Tensor = torch.sum(chunk_label, dim=-1)
        pred_label_sum: torch.Tensor = torch.sum(predictions[..., 1], dim=-1)
        diff: torch.Tensor = torch.abs(pred_label_sum - true_label_sum).type(torch.float)
        return ModelLoss(chunker_loss=loss_ignore + self.loss_lambda * torch.mean(diff) * self.mode)

    def update_metrics(self, model_out: ModelOutput) -> None:
        b: Batch = model_out.batch
        for pred, aspect, opinion in zip(model_out.predicted_spans, b.aspect_spans, b.opinion_spans):
            true: torch.Tensor = torch.cat([aspect, opinion], dim=0)
            self.metrics(pred, true)

    def get_metrics(self) -> ModelMetric:
        return ModelMetric(chunker_metric=self.metrics.compute())

    def reset_metrics(self) -> None:
        self.metrics.reset()
