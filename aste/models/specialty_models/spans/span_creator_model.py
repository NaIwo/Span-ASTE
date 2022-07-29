from ASTE.utils import config
from ASTE.dataset.reader import Batch
from ASTE.dataset.domain.const import ChunkCode
from ASTE.aste.models.specialty_models.spans.crf import CRF
from ASTE.aste.tools.metrics import Metric, get_selected_metrics
from ASTE.aste.models import ModelOutput, ModelLoss, ModelMetric, BaseModel

import torch
from typing import List, Tuple
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SpanCreatorModel(BaseModel):
    def __init__(self, input_dim: int, model_name: str = 'Span Creator Model'):
        super(SpanCreatorModel, self).__init__(model_name)
        self.metrics: Metric = Metric(name='Span Creator', metrics=get_selected_metrics(for_spans=True)).to(
            config['general']['device'])

        self.input_dim: int = input_dim

        self.crf = CRF(num_tags=3, batch_first=True)
        self.lstm = torch.nn.LSTM(input_dim, input_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.linear_layer = torch.nn.Linear(input_dim, input_dim // 2)
        self.final_layer = torch.nn.Linear(input_dim // 2, 3)

    def forward(self, data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.get_features(data, mask)

    def get_features(self, data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths: torch.Tensor = torch.sum(mask, dim=1).cpu()
        hidden: Tuple[torch.Tensor, torch.Tensor] = self.init_hidden(self.input_dim, len(lengths))

        embeddings = pack_padded_sequence(data, lengths, batch_first=True)
        context, _ = self.lstm(embeddings, hidden)
        context, _ = pad_packed_sequence(context, batch_first=True)

        context = self.linear_layer(context)
        return self.final_layer(context)

    @staticmethod
    def init_hidden(size: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.randn(2, batch_size, size // 2).to(config['general']['device']),
                torch.randn(2, batch_size, size // 2).to(config['general']['device']))

    def get_spans(self, data: torch.Tensor, batch: Batch) -> List[torch.Tensor]:
        results: List[torch.Tensor] = list()
        best_paths: List[List[int]] = self.crf.decode(data, mask=batch.mask[:, :data.shape[1], ...])

        for best_path, sample in zip(best_paths, batch):
            best_path = torch.tensor(best_path).to(config['general']['device'])
            offset: int = sample.sentence_obj[0].encoder.offset
            best_path[:offset] = ChunkCode.NOT_SPLIT
            best_path[sum(sample.mask[0]) - offset:] = ChunkCode.NOT_SPLIT
            results.append(self.get_spans_from_sequence(best_path, sample))

        return results

    @staticmethod
    def get_spans_from_sequence(seq: torch.Tensor, sample: Batch) -> torch.Tensor:
        begins: torch.Tensor = torch.where(seq == ChunkCode.BEGIN_SPLIT)[0]
        begins = torch.cat((begins, torch.tensor([len(seq)], device=config['general']['device'])))
        begins: List = [sample.sentence_obj[0].agree_index(idx) for idx in begins]
        results: List = list()

        idx: int
        b_idx: int
        end_idx: int
        for idx, b_idx in enumerate(begins[:-1]):
            s: torch.Tensor = seq[b_idx:begins[idx + 1]]
            if ChunkCode.NOT_SPLIT in s:
                end_idx = int(torch.where(s == ChunkCode.NOT_SPLIT)[0][0])
                end_idx += b_idx - 1
            else:
                end_idx = begins[idx + 1] - 1

            results.append([b_idx, end_idx])

        if not results:
            results.append([0, len(seq) - 1])
        return torch.tensor(results).to(config['general']['device'])

    def get_loss(self, model_out: ModelOutput) -> ModelLoss:
        loss = -self.crf(model_out.chunker_output, model_out.batch.chunk_label, model_out.batch.mask,
                         reduction='token_mean')
        return ModelLoss(chunker_loss=loss)

    def update_metrics(self, model_out: ModelOutput) -> None:
        b: Batch = model_out.batch
        for pred, aspect, opinion in zip(model_out.predicted_spans, b.aspect_spans, b.opinion_spans):
            true: torch.Tensor = torch.cat([aspect, opinion], dim=0).unique(dim=0)
            true_count: int = true.shape[0] - int(-1 in true)
            self.metrics(pred, true, true_count)

    def get_metrics(self) -> ModelMetric:
        return ModelMetric(chunker_metric=self.metrics.compute())

    def reset_metrics(self) -> None:
        self.metrics.reset()
