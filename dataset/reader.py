from .domain import Sentence, get_chunk_label_from_sentence
from ASTE.utils import config
from ASTE.dataset.domain.const import ChunkCode
from ASTE.dataset.encoders import BaseEncoder, BertEncoder

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Union
import os
from collections import Iterable


class ASTEDataset(Dataset):
    def __init__(self, data_path: Union[str, List[str]], encoder: BaseEncoder = BertEncoder()):
        self.sentences: List[Sentence] = list()

        if isinstance(data_path, str):
            data_path = [data_path]

        data_p: str
        for data_p in data_path:
            self.add_sentences(data_p, encoder)

    def add_sentences(self, data_path, encoder):
        with open(data_path, 'r') as file:
            line: str
            for line in tqdm(file.readlines(), desc=f'Loading data from path: {data_path}'):
                sentence = Sentence(line, encoder)
                self.sentences.append(sentence)

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

    def __iter__(self):
        self.num: int = -1
        return self

    def __next__(self):
        self.num += 1
        if self.num >= len(self):
            raise StopIteration
        return self.sentences[self.num]


class DatasetLoader:
    def __init__(self, data_path: str):
        self.data_path: str = data_path

    def load(self, name: Union[str, List[str]], encoder: BaseEncoder = BertEncoder()) -> DataLoader:
        if isinstance(name, str):
            name = [name]

        paths: List = list()
        data_name: str
        for data_name in name:
            paths.append(os.path.join(self.data_path, data_name))
        dataset: ASTEDataset = ASTEDataset(paths, encoder)
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
        sample: Sentence
        for sample in batch:
            sentence_objs.append(sample)
            encoded_sentences.append(torch.tensor(sample.encoded_sentence))
            aspect_spans.append(torch.tensor(sample.get_aspect_spans()))
            opinion_spans.append(torch.tensor(sample.get_opinion_spans()))
            chunk_labels.append(torch.tensor(get_chunk_label_from_sentence(sample)))
            sub_words_masks.append(torch.tensor(sample.sub_words_mask))
            lengths.append(sample.encoded_sentence_length)

        lengths: torch.Tensor = torch.tensor(lengths, dtype=torch.int64)
        sentence_batch = pad_sequence(encoded_sentences, padding_value=0, batch_first=True)
        aspect_spans_batch = pad_sequence(aspect_spans, padding_value=-1, batch_first=True)
        opinion_spans_batch = pad_sequence(opinion_spans, padding_value=-1, batch_first=True)
        chunk_batch = pad_sequence(chunk_labels, padding_value=ChunkCode.NOT_RELEVANT, batch_first=True)
        sub_words_masks_batch = pad_sequence(sub_words_masks, padding_value=0, batch_first=True)
        mask: torch.Tensor = self._construct_mask(lengths)
        idx: torch.Tensor = torch.argsort(lengths, descending=True)

        sentence_obj = self._get_list_of_sentence_objs(sentence_objs, idx)
        return Batch(sentence_obj=sentence_obj,
                     sentence=sentence_batch[idx].to(config['general']['device']),
                     aspect_spans=aspect_spans_batch[idx].to(config['general']['device']),
                     opinion_spans=opinion_spans_batch[idx].to(config['general']['device']),
                     chunk_label=chunk_batch[idx].to(config['general']['device']),
                     sub_words_masks=sub_words_masks_batch[idx].to(config['general']['device']),
                     mask=mask[idx].to(config['general']['device']))

    @staticmethod
    def _get_list_of_sentence_objs(sentence_objs: List[Sentence], idx: torch.Tensor) -> List[Sentence]:
        sentence_obj = np.array(sentence_objs)[idx]
        if not isinstance(sentence_obj, Iterable):
            sentence_obj = [sentence_obj]
        else:
            sentence_obj = list(sentence_obj)
        return sentence_obj

    @staticmethod
    def _construct_mask(lengths: torch.Tensor) -> torch.Tensor:
        max_len: int = max(lengths).item()
        mask: torch.Tensor = torch.arange(max_len).expand(lengths.shape[0], max_len)
        return (mask < lengths.unsqueeze(1)).type(torch.int8)


class Batch:
    def __init__(self, *, sentence_obj: [Sentence], sentence: torch.Tensor,
                 aspect_spans: torch.Tensor, opinion_spans: torch.Tensor,
                 chunk_label: torch.Tensor, sub_words_masks: torch.Tensor, mask: torch.Tensor):
        self.sentence_obj: List[Sentence] = sentence_obj
        self.sentence: torch.Tensor = sentence
        self.aspect_spans: torch.Tensor = aspect_spans
        self.opinion_spans: torch.Tensor = opinion_spans
        self.chunk_label: torch.Tensor = chunk_label
        self.sub_words_mask: torch.Tensor = sub_words_masks
        self.mask: torch.Tensor = mask

    @classmethod
    def from_sentence(cls, sentence: Sentence):
        return cls(
            sentence_obj=[sentence],
            sentence=torch.tensor([sentence.encoded_sentence]).to(config['general']['device']),
            sub_words_masks=torch.tensor([sentence.sub_words_mask]).to(config['general']['device']),
            mask=torch.ones(size=(1, len(sentence.encoded_sentence))).to(config['general']['device']),
            aspect_spans=torch.tensor([[]]),
            opinion_spans=torch.tensor([[]]),
            chunk_label=torch.tensor([[]])
        )

    def __iter__(self):
        self.num: int = -1
        return self

    def __next__(self):
        self.num += 1
        if self.num >= len(self.sentence_obj):
            raise StopIteration
        return Batch(sentence_obj=[self.sentence_obj[self.num]],
                     sentence=self.sentence[self.num].unsqueeze(0),
                     aspect_spans=self.aspect_spans[self.num].unsqueeze(0),
                     opinion_spans=self.opinion_spans[self.num].unsqueeze(0),
                     chunk_label=self.chunk_label[self.num].unsqueeze(0),
                     sub_words_masks=self.sub_words_mask[self.num].unsqueeze(0),
                     mask=self.mask[self.num].unsqueeze(0))

    def __len__(self) -> int:
        return len(self.sentence_obj)
