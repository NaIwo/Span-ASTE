from ASTE.aste.models.models import BaseModel
from ASTE.utils import config
from ASTE.dataset.reader import Batch
from ASTE.aste.models import ModelOutput, ModelLoss, ModelMetric

import torch
from typing import List


class SpanModel(BaseModel):
    def __init__(self, model_name: str = 'Chunker Model', *args, **kwargs):
        super(SpanModel, self).__init__(model_name)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return data

    def get_spans_from_batch_prediction(self, batch: Batch, *args, **kwargs) -> List[torch.Tensor]:
        results: List[torch.Tensor] = list()

        sample: Batch
        prediction: torch.Tensor
        for sample in batch:
            predicted_spans: torch.Tensor = self._get_spans_from_single_sample(sample)
            results.append(predicted_spans)
        return results

    @staticmethod
    def _get_spans_from_single_sample(sample: Batch) -> torch.Tensor:
        spans: List = list()
        for start_idx in range(sample.sentence_obj[0].sentence_length):
            for end_idx in range(start_idx, start_idx+3):
                if end_idx >= sample.sentence_obj[0].sentence_length:
                    break
                spans.append([sample.sentence_obj[0].get_index_after_encoding(start_idx),
                              sample.sentence_obj[0].get_index_after_encoding(end_idx)])
        return torch.tensor(spans, device=config['general']['device'])
        # results = torch.cat([sample.aspect_spans[0], sample.opinion_spans[0]], dim=0).unique(dim=0)
        # results = results[torch.where(results != -1)]
        # return results.reshape(-1, 2)

    def get_loss(self, model_out: ModelOutput) -> ModelLoss:
        return ModelLoss(chunker_loss=torch.tensor(0.).to(config['general']['device']))

    def update_metrics(self, model_out: ModelOutput) -> None:
        pass

    def get_metrics(self) -> ModelMetric:
        return ModelMetric(chunker_metric={})

    def reset_metrics(self) -> None:
        pass
