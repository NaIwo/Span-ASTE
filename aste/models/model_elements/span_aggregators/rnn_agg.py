from typing import List

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ASTE.utils import config
from .base_agg import BaseAggregator


class RnnAggregator(BaseAggregator, Module):
    def __init__(self, input_dim: int,
                 bidirectional: bool = False,
                 num_layers: int = 2,
                 model_name: str = 'RNN Aggregator', *args, **kwargs):
        Module.__init__(self)
        BaseAggregator.__init__(self, input_dim=input_dim, model_name=model_name)
        self._out_dim: int = input_dim

        self.bidirectional: bool = bidirectional
        self.rnn = torch.nn.GRU(input_dim, input_dim // (1 + int(bidirectional)),
                                num_layers=num_layers,
                                bidirectional=self.bidirectional,
                                batch_first=True)

    @property
    def output_dim(self):
        return self._out_dim

    def _get_agg_sentence_embeddings(self, sentence_embeddings: Tensor, sentence_spans: Tensor) -> Tensor:
        span_embeddings: List[Tensor] = list()
        lengths: List[Tensor] = list()
        span: Tensor
        for span in sentence_spans:
            span_emb: Tensor = sentence_embeddings[span[0]:span[1] + 1]
            lengths.append((span[1] + 1) - span[0])
            span_embeddings.append(span_emb)

        return self._aggregate_spans(span_embeddings, lengths)

    def _aggregate_spans(self, span_embeddings: List[Tensor], lengths: List[Tensor]) -> Tensor:
        lengths: Tensor = torch.tensor(lengths)
        lengths, perm_index = lengths.sort(0, descending=True)

        data: Tensor = self.pad_sequence(span_embeddings)
        data = data[perm_index]

        hidden: Tensor = self._init_hidden(self.input_dim, len(span_embeddings))

        embeddings = pack_padded_sequence(data, lengths, batch_first=True)
        packed_output: PackedSequence = self.rnn(embeddings, hidden)[0]
        output: Tensor = pad_packed_sequence(packed_output, batch_first=True)[0]
        lengths = (lengths - 1).to(config['general']['device'])
        lengths = lengths.view(-1, 1).unsqueeze(1).expand(*data.shape)
        output = torch.gather(output, dim=1, index=lengths)[:, 0, ...]
        output = output[perm_index.argsort()]

        return output

    def _init_hidden(self, size: int, batch_size: int) -> Tensor:
        bi_int: int = 1 + int(self.bidirectional)
        size: int = size // bi_int
        first_dim: int = self.rnn.num_layers * bi_int
        empty_tensor: Tensor = torch.empty(first_dim, batch_size, size).to(config['general']['device'])
        return torch.nn.init.xavier_uniform_(empty_tensor)
