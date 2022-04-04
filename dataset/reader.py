from .domain.sentence import Sentence

from torch.utils.data import Dataset
from typing import List
import os

# TODO torch dataset


class ASTEDataset(Dataset):
    def __init__(self, data_path: str):
        self.sentences: List[Sentence] = list()

        with open(data_path, 'r') as file:
            line: str
            for line in file.readlines():
                self.sentences.append(Sentence(line.strip()))

    def __str__(self) -> str:
        return str({
            'Number of sentences': len(self.sentences)
        })

    def __repr__(self):
        return self.__str__()


class DataReader:
    def __init__(self, data_path: str):
        self.data_path: str = data_path

    def read(self, name: str) -> ASTEDataset:
        return ASTEDataset(os.path.join(self.data_path, name))
