from ASTE.utils import config
from ASTE.dataset.reader import Batch
from ASTE.aste.models import ModelOutput, ModelLoss, ModelMetric, BaseModel
from ASTE.aste.tools.metrics import Metric, get_selected_metrics

import torch
from typing import List
from functools import lru_cache


class AllSpansCreatorModel(BaseModel):
    def __init__(self, model_name: str = 'All Spans Creator Model', *args, **kwargs):
        super(AllSpansCreatorModel, self).__init__(model_name)
        self.window_size: int = 6

        self.metrics: Metric = Metric(name='Span Creator', metrics=get_selected_metrics(for_spans=True)).to(
            config['general']['device'])

    def forward(self, *args, **kwargs) -> None:
        pass

    def get_spans(self, _, batch: Batch) -> List[torch.Tensor]:
        results: List = list()
        sample: Batch
        for sample in batch:
            result: torch.Tensor = self.get_spans_from_sample(sample)
            results.append(result)
        return results

    @lru_cache(maxsize=2048)
    def get_spans_from_sample(self, sample: Batch) -> torch.Tensor:
        results: List = list()
        offset: int = sample.sentence_obj[0].encoder.offset
        start_idx: int
        for start_idx in range(offset, sample.sub_words_mask.shape[1] - offset):
            if sample.sub_words_mask[:, start_idx] == 0:
                continue
            result: List = list()
            window: int = 1
            window_size: int = self.window_size
            while (window <= window_size) and (start_idx + window < sample.sub_words_mask.shape[1]):
                if sample.sub_words_mask[:, start_idx + window] == 0:
                    window_size += 1
                else:
                    result.append([start_idx, start_idx + window - 1])
                window += 1
            results += result
        return torch.tensor(results, device=config['general']['device'])

    def get_loss(self, model_out: ModelOutput) -> ModelLoss:
        return ModelLoss()

    def update_metrics(self, model_out: ModelOutput) -> None:
        b: Batch = model_out.batch
        for pred, aspect, opinion in zip(model_out.predicted_spans, b.aspect_spans, b.opinion_spans):
            true: torch.Tensor = torch.cat([aspect, opinion], dim=0).unique(dim=0)
            true_count: int = true.shape[0] - int(-1 in true)
            self.metrics(pred, true, true_count)

    def get_metrics(self) -> ModelMetric:
        return ModelMetric(span_creator_metric=self.metrics.compute())

    def reset_metrics(self) -> None:
        self.metrics.reset()