from typing import List, Optional, Callable

import torch

from ASTE.aste.models import ModelOutput, ModelLoss, ModelMetric, BaseModel
from ASTE.aste.models.specialty_models.spans.crf import CRF
from ASTE.aste.tools.metrics import Metric, get_selected_metrics
from ASTE.dataset.domain.const import SpanCode
from ASTE.dataset.reader import Batch
from ASTE.utils import config


class SpanCreatorModel(BaseModel):
    def __init__(self, input_dim: int, model_name: str = 'Span Creator Model',
                 extend_spans: Optional[List[int]] = None):
        super(SpanCreatorModel, self).__init__(model_name)
        self.metrics: Metric = Metric(name='Span Creator', metrics=get_selected_metrics(for_spans=True)).to(
            config['general']['device'])

        self.extend_spans: Optional[List[int]] = extend_spans
        if extend_spans is None:
            self.extend_spans: List[int] = [-1, 1]

        self.input_dim: int = input_dim

        self.crf = CRF(num_tags=3, batch_first=True)
        self.linear_layer = torch.nn.Linear(input_dim, input_dim // 2)
        self.final_layer = torch.nn.Linear(input_dim // 2, 3)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.get_features(data)

    def get_features(self, data: torch.Tensor) -> torch.Tensor:
        out = self.linear_layer(data)
        return self.final_layer(out)

    def get_spans(self, data: torch.Tensor, batch: Batch) -> List[torch.Tensor]:
        results: List[torch.Tensor] = list()
        best_paths: List[List[int]] = self.crf.decode(data, mask=batch.emb_mask[:, :data.shape[1], ...])

        for best_path, sample in zip(best_paths, batch):
            best_path = torch.tensor(best_path).to(config['general']['device'])
            offset: int = sample.sentence_obj[0].encoder.offset
            best_path[:offset] = SpanCode.NOT_SPLIT
            best_path[sum(sample.emb_mask[0]) - offset:] = SpanCode.NOT_SPLIT
            results.append(self.get_spans_from_sequence(best_path, sample))

        return results

    def get_spans_from_sequence(self, seq: torch.Tensor, sample: Batch) -> torch.Tensor:
        begins: torch.Tensor = torch.where(seq == SpanCode.BEGIN_SPLIT)[0]
        begins = torch.cat((begins, torch.tensor([len(seq)], device=config['general']['device'])))
        begins: List = [sample.sentence_obj[0].agree_index(idx) for idx in begins]
        results: List[List[int, int]] = list()

        idx: int
        b_idx: int
        end_idx: int
        for idx, b_idx in enumerate(begins[:-1]):
            s: torch.Tensor = seq[b_idx:begins[idx + 1]]
            if SpanCode.NOT_SPLIT in s:
                end_idx = int(torch.where(s == SpanCode.NOT_SPLIT)[0][0])
                end_idx += b_idx - 1
            else:
                end_idx = begins[idx + 1] - 1

            if (end_idx >= b_idx) and ([b_idx, end_idx] not in results):
                results.append([b_idx, end_idx])

        if not results:
            results.append([0, len(seq) - 1])
        else:
            results = self.extend_results(results, sample)
        return torch.tensor(results).to(config['general']['device'])

    def extend_results(self, results: List, sample: Batch) -> List:
        extended_results: List = results[:]
        before: Callable = sample.sentence_obj[0].get_index_before_encoding
        after: Callable = sample.sentence_obj[0].get_index_after_encoding

        begin: int
        end: int
        for begin, end in results:
            temp: List = [[after(before(begin) + shift), end] for shift in self.extend_spans]
            self._add_correct_extended(extended_results, temp)

            temp: List = [[begin, after(before(end) + shift)] for shift in self.extend_spans]
            self._add_correct_extended(extended_results, temp)

        return extended_results

    def _add_correct_extended(self, results: List, extended: List) -> List:
        begin: int
        end: int
        for begin, end in extended:
            if -1 != end >= begin != -1 and [begin, end] not in results:
                results.append([begin, end])

        return results

    def get_loss(self, model_out: ModelOutput) -> ModelLoss:
        loss = -self.crf(model_out.span_creator_output, model_out.batch.chunk_label, model_out.batch.emb_mask,
                         reduction='token_mean')
        return ModelLoss(span_creator_loss=loss)

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
