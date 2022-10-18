import os
import warnings
from collections import Iterable
from typing import List, Union, TypeVar

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from ASTE.dataset.domain.const import SpanCode
from ASTE.dataset.encoders import BaseEncoder, BertEncoder
from ASTE.utils import config
from .domain import Sentence, get_span_label_from_sentence

ASTE = TypeVar('ASTE', bound='ASTEDataset')


class ASTEDataset(Dataset):
    def __init__(self, data_path: Union[str, List[str]],
                 encoder: BaseEncoder = BertEncoder(),
                 include_sub_words_info_in_mask: bool = True):
        self.sentences: List[Sentence] = list()

        if isinstance(data_path, str):
            data_path = [data_path]

        data_p: str
        for data_p in data_path:
            self.add_sentences(data_p, encoder, include_sub_words_info_in_mask)

    def add_sentences(self, data_path: str, encoder: BaseEncoder, include_sub_words_info_in_mask: bool):
        with open(data_path, 'r') as file:
            line: str
            for line in tqdm(file.readlines(), desc=f'Loading data from path: {data_path}'):
                sentence = Sentence(line, encoder, include_sub_words_info_in_mask)
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

    def sort(self) -> None:
        self.sentences.sort(key=lambda sentence: sentence.sentence)


class DatasetLoader:
    def __init__(self, data_path: str,
                 encoder: BaseEncoder = BertEncoder(),
                 include_sub_words_info_in_mask: bool = True):

        self.data_path: str = data_path
        self.encoder: BaseEncoder = encoder
        self.include_sub_words_info_in_mask: bool = include_sub_words_info_in_mask

        if not include_sub_words_info_in_mask:
            warnings.warn('Careful! If you do not want to include sub-word elements in '
                          'the mask make sure that the embedding model also take this into account '
                          '(aggregate immersions from sub-words or do not generate such situations)! ')

    def load(self, name: Union[str, List[str]]) -> DataLoader:
        if isinstance(name, str):
            name = [name]

        paths: List = list()
        data_name: str
        for data_name in name:
            paths.append(os.path.join(self.data_path, data_name))
        dataset: ASTEDataset = ASTEDataset(paths, self.encoder, self.include_sub_words_info_in_mask)
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
        emb_lengths: List = list()
        sample: Sentence
        for sample in batch:
            sentence_objs.append(sample)
            encoded_sentences.append(torch.tensor(sample.encoded_sentence))
            aspect_spans.append(torch.tensor(sample.get_aspect_spans()))
            opinion_spans.append(torch.tensor(sample.get_opinion_spans()))
            chunk_labels.append(torch.tensor(get_span_label_from_sentence(sample)))
            sub_words_masks.append(torch.tensor(sample.get_sub_words_mask()))
            lengths.append(sample.encoded_sentence_length)
            emb_lengths.append(sample.emb_sentence_length)

        lengths: Tensor = torch.tensor(lengths, dtype=torch.int64)
        emb_lengths: Tensor = torch.tensor(emb_lengths, dtype=torch.int64)
        sentence_batch = pad_sequence(encoded_sentences, padding_value=0, batch_first=True)
        aspect_spans_batch = pad_sequence(aspect_spans, padding_value=-1, batch_first=True)
        opinion_spans_batch = pad_sequence(opinion_spans, padding_value=-1, batch_first=True)
        chunk_batch = pad_sequence(chunk_labels, padding_value=SpanCode.NOT_RELEVANT, batch_first=True)
        sub_words_masks_batch = pad_sequence(sub_words_masks, padding_value=0, batch_first=True)
        mask: Tensor = self._construct_mask(lengths)
        emb_mask: Tensor = self._construct_mask(emb_lengths)
        idx: Tensor = torch.argsort(lengths, descending=True)

        sentence_obj = self._get_list_of_sentence_objs(sentence_objs, idx)
        return Batch(sentence_obj=sentence_obj,
                     sentence=sentence_batch[idx].to(config['general']['device']),
                     aspect_spans=aspect_spans_batch[idx].to(config['general']['device']),
                     opinion_spans=opinion_spans_batch[idx].to(config['general']['device']),
                     chunk_label=chunk_batch[idx].to(config['general']['device']),
                     sub_words_masks=sub_words_masks_batch[idx].to(config['general']['device']),
                     mask=mask[idx].to(config['general']['device']),
                     emb_mask=emb_mask[idx].to(config['general']['device']))

    @staticmethod
    def _get_list_of_sentence_objs(sentence_objs: List[Sentence], idx: Tensor) -> List[Sentence]:
        sentence_obj = np.array(sentence_objs)[idx]
        if not isinstance(sentence_obj, Iterable):
            sentence_obj = [sentence_obj]
        else:
            sentence_obj = list(sentence_obj)
        return sentence_obj

    @staticmethod
    def _construct_mask(lengths: Tensor) -> Tensor:
        max_len: int = max(lengths).item()
        mask: Tensor = torch.arange(max_len).expand(lengths.shape[0], max_len)
        return (mask < lengths.unsqueeze(1)).type(torch.int8)


class Batch:
    def __init__(self, *, sentence_obj: [Sentence], sentence: Tensor,
                 aspect_spans: Tensor, opinion_spans: Tensor,
                 chunk_label: Tensor, sub_words_masks: Tensor, mask: Tensor, emb_mask: Tensor):
        self.sentence_obj: List[Sentence] = sentence_obj
        self.sentence: Tensor = sentence
        self.aspect_spans: Tensor = aspect_spans
        self.opinion_spans: Tensor = opinion_spans
        self.chunk_label: Tensor = chunk_label
        self.sub_words_mask: Tensor = sub_words_masks
        self.mask: Tensor = mask
        self.emb_mask: Tensor = emb_mask

    @classmethod
    def from_sentence(cls, sentence: Sentence):
        return cls(
            sentence_obj=[sentence],
            sentence=torch.tensor([sentence.encoded_sentence]).to(config['general']['device']),
            sub_words_masks=torch.tensor([sentence.get_sub_words_mask()]).to(config['general']['device']),
            mask=torch.ones(size=(1, sentence.encoded_sentence_length)).to(config['general']['device']),
            emb_mask=torch.ones(size=(1, sentence.emb_sentence_length)).to(config['general']['device']),
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
                     mask=self.mask[self.num].unsqueeze(0),
                     emb_mask=self.emb_mask[self.num].unsqueeze(0))

    def __len__(self) -> int:
        return len(self.sentence_obj)
