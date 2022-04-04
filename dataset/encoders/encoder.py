from ASTE.dataset.encoders.bert_encoder import BertEncoder
from ASTE.dataset.encoders.base_encoder import BaseEncoder
from ASTE.utils import config

from typing import List


class Encoder:
    def __init__(self, encoder: BaseEncoder):
        self._encoder: BaseEncoder = encoder
        self.offset: int = self._encoder.offset

    def encode(self, sentence: str) -> List:
        return self._encoder.encode(sentence=sentence)

    def encode_single_word(self, word: str) -> List:
        return self._encoder.encode_single_word(word)

    def encode_word_by_word(self, sentence: str) -> List[List]:
        encoded_words: List = list()
        word: str
        for word in sentence.strip().split():
            encoded_words.append(self.encode_single_word(word))

        return encoded_words


if config['encoder']['type'] == 'bert':
    encoder: Encoder = Encoder(BertEncoder())
else:
    encoder: Encoder = Encoder(BaseEncoder())
