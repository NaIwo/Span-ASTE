from .domain import Sentence, Label
from ASTE.utils import config
from ASTE.dataset.domain.const import ChunkCode
from ASTE.dataset.encoders import BaseEncoder

import torch
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

    @staticmethod
    def _collate_fn(batch: List):

        class Batch:
            def __init__(self, sentence: torch.Tensor, chunk_label: torch.Tensor, mask: torch.Tensor):
                self.sentence: torch.Tensor = sentence
                self.chunk_label: torch.Tensor = chunk_label
                self.mask: torch.Tensor = mask

        encoded_sentences: List = list()
        chunk_labels: List = list()
        lengths: List = list()
        sample: Dict
        for sample in batch:
            encoded_sentences.append(torch.tensor(sample["sentence"].encoded_sentence))
            chunk_labels.append(torch.tensor(sample["chunk"]))
            lengths.append(sample["sentence"].encoded_sentence_length)

        sentence_batch = pad_sequence(encoded_sentences, padding_value=0, batch_first=True)
        chunk_batch = pad_sequence(chunk_labels, padding_value=int(ChunkCode.NOT_RELEVANT), batch_first=True)
        max_len: int = max(lengths)
        mask: torch.Tensor = torch.arange(max_len).expand(len(lengths), max_len)
        lengths: torch.Tensor = torch.tensor(lengths, dtype=torch.int64)
        mask = mask < lengths.unsqueeze(1)
        idx = torch.argsort(lengths, descending=True)

        return Batch(sentence_batch[idx].to(config['general']['device']), chunk_batch[idx].to(
            config['general']['device']), mask.to(config['general']['device'])[idx].type(torch.int8))
