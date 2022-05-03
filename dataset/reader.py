from .domain import Sentence, get_label_from_sentence
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
        self.chunk_labels: List[np.ndarray] = list()

        with open(data_path, 'r') as file:
            line: str
            for line in file.readlines():
                sentence = Sentence(line.strip(), encoder)
                self.sentences.append(sentence)
                self.chunk_labels.append(get_label_from_sentence(sentence))

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx):
        return {
            'sentence': self.sentences[idx],
            'chunk': self.chunk_labels[idx]
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
        aspect_spans: List = list()
        opinion_spans: List = list()
        chunk_labels: List = list()
        sub_words_masks: List = list()
        lengths: List = list()
        sample: Dict
        for sample in batch:
            sentence_objs.append(sample["sentence"])
            encoded_sentences.append(torch.tensor(sample["sentence"].encoded_sentence))
            aspect_spans.append(torch.tensor(sample["sentence"].get_aspect_spans()))
            opinion_spans.append(torch.tensor(sample["sentence"].get_opinion_spans()))
            chunk_labels.append(torch.tensor(sample["chunk"]))
            sub_words_masks.append(torch.tensor(sample["sentence"].sub_words_mask))
            lengths.append(sample["sentence"].encoded_sentence_length)

        lengths: torch.Tensor = torch.tensor(lengths, dtype=torch.int64)
        sentence_batch = pad_sequence(encoded_sentences, padding_value=0, batch_first=True)
        aspect_spans_batch = pad_sequence(aspect_spans, padding_value=-1, batch_first=True)
        opinion_spans_batch = pad_sequence(opinion_spans, padding_value=-1, batch_first=True)
        chunk_batch = pad_sequence(chunk_labels, padding_value=ChunkCode.NOT_RELEVANT, batch_first=True)
        sub_words_masks_batch = pad_sequence(sub_words_masks, padding_value=0, batch_first=True)
        mask = self._construct_mask(lengths)
        idx = torch.argsort(lengths, descending=True)

        return Batch(sentence_obj=list(np.array(sentence_objs)[idx]),
                     sentence=sentence_batch[idx].to(config['general']['device']),
                     aspect_spans=aspect_spans_batch[idx].to(config['general']['device']),
                     opinion_spans=opinion_spans_batch[idx].to(config['general']['device']),
                     chunk_label=chunk_batch[idx].to(config['general']['device']),
                     sub_words_masks=sub_words_masks_batch[idx].to(config['general']['device']),
                     mask=mask[idx].to(config['general']['device']))

    @staticmethod
    def _construct_mask(lengths: torch.Tensor) -> torch.Tensor:
        max_len: int = max(lengths).item()
        mask: torch.Tensor = torch.arange(max_len).expand(lengths.shape[0], max_len)
        return (mask < lengths.unsqueeze(1)).type(torch.int8)


class Batch:
    def __init__(self, sentence_obj: [Sentence], sentence: torch.Tensor,
                 aspect_spans: torch.Tensor, opinion_spans: torch.Tensor,
                 chunk_label: torch.Tensor, sub_words_masks: torch.Tensor, mask: torch.Tensor):
        self.sentence_obj: List[Sentence] = sentence_obj
        self.sentence: torch.Tensor = sentence
        self.aspect_spans: torch.Tensor = aspect_spans
        self.opinion_spans: torch.Tensor = opinion_spans
        self.chunk_label: torch.Tensor = chunk_label
        self.sub_words_mask: torch.Tensor = sub_words_masks
        self.mask: torch.Tensor = mask

    def __iter__(self):
        self.num: int = -1
        return self

    def __next__(self):
        self.num += 1
        if self.num >= len(self.sentence_obj):
            raise StopIteration
        return Batch([self.sentence_obj[self.num]],
                     self.sentence[self.num].unsqueeze(0),
                     self.aspect_spans[self.num].unsqueeze(0),
                     self.opinion_spans[self.num].unsqueeze(0),
                     self.chunk_label[self.num].unsqueeze(0),
                     self.sub_words_mask[self.num].unsqueeze(0),
                     self.mask[self.num].unsqueeze(0))
