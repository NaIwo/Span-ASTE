from .domain import Sentence, Label
from ASTE.utils import config
from ASTE.dataset.domain.const import ChunkCode
from ASTE.dataset.encoders import BaseEncoder

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict
import os


class ASTEDataset(Dataset):
    def __init__(self, data_path: str, encoder):
        self.sentences: List[Sentence] = list()
        self.chunk_labels: List[Label] = list()

        with open(data_path, 'r') as file:
            line: str
            for line in file.readlines():
                sentence = Sentence(line.strip(), encoder)
                self.sentences.append(sentence)
                self.chunk_labels.append(Label.from_sentence(sentence))

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx):
        return {
            'sentence': self.sentences[idx],
            'chunk': self.chunk_labels[idx].chunk
        }


class DatasetLoader:
    def __init__(self, data_path: str):
        self.data_path: str = data_path

    def load(self, name: str, encoder: BaseEncoder) -> DataLoader:
        dataset: ASTEDataset = ASTEDataset(os.path.join(self.data_path, name), encoder)
        return DataLoader(dataset, batch_size=config['dataset']['batch-size'], shuffle=True, prefetch_factor=2,
                          collate_fn=self._collate_fn)

    def _collate_fn(self, batch: List):
        sentence_objs: List[Sentence] = list()
        encoded_sentences: List = list()
        chunk_labels: List = list()
        lengths: List = list()
        sample: Dict
        for sample in batch:
            sentence_objs.append(sample["sentence"])
            encoded_sentences.append(torch.tensor(sample["sentence"].encoded_sentence))
            chunk_labels.append(torch.tensor(sample["chunk"]))
            lengths.append(sample["sentence"].encoded_sentence_length)

        lengths: torch.Tensor = torch.tensor(lengths, dtype=torch.int64)
        sentence_batch = pad_sequence(encoded_sentences, padding_value=0, batch_first=True)
        chunk_batch = pad_sequence(chunk_labels, padding_value=int(ChunkCode.NOT_RELEVANT), batch_first=True)
        mask = self._construct_mask(lengths)
        idx = torch.argsort(lengths, descending=True)

        return _Batch(sentence_obj=list(np.array(sentence_objs)[idx]),
                      sentence=sentence_batch[idx].to(config['general']['device']),
                      chunk_label=chunk_batch[idx].to(config['general']['device']),
                      mask=mask.to(config['general']['device'])[idx].type(torch.int8))

    @staticmethod
    def _construct_mask(lengths) -> torch.Tensor:
        max_len: int = max(lengths).item()
        mask: torch.Tensor = torch.arange(max_len).expand(lengths.shape[0], max_len)
        return mask < lengths.unsqueeze(1)


class _Batch:
    def __init__(self, sentence_obj: [Sentence], sentence: torch.Tensor, chunk_label: torch.Tensor, mask: torch.Tensor):
        self.sentence_obj: List[Sentence] = sentence_obj
        self.sentence: torch.Tensor = sentence
        self.chunk_label: torch.Tensor = chunk_label
        self.mask: torch.Tensor = mask

    def __iter__(self):
        self.num: int = -1
        return self

    def __next__(self):
        self.num += 1
        if self.num >= len(self.sentence_obj):
            raise StopIteration
        return _Batch([self.sentence_obj[self.num]],
                      self.sentence[self.num].unsqueeze(0),
                      self.chunk_label[self.num].unsqueeze(0),
                      self.mask[self.num].unsqueeze(0))
