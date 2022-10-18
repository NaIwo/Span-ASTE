from .base_encoder import BaseEncoder
from ASTE.utils import config
from typing import List

from transformers import BertTokenizer


class BertEncoder(BaseEncoder):
    def __init__(self):
        super().__init__(encoder_name='bert tokenizer')
        self.encoder: BertTokenizer = BertTokenizer.from_pretrained(config['encoder']['bert']['source'])
        self.offset: int = 1

    def encode(self, sentence: str) -> List:
        return self.encoder.encode(sentence)

    def encode_single_word(self, word: str) -> List:
        return self.encoder.encode(word, add_special_tokens=False)


class BertEncoderWithoutSubwords(BaseEncoder):
    def __init__(self):
        super().__init__(encoder_name='bert without subwords tokenizer')
        self.encoder: BertTokenizer = BertTokenizer.from_pretrained(config['encoder']['bert']['source'])
        self.offset: int = 1

    def encode(self, sentence: str) -> List:
        special_tokens: List = self.encoder.encode(sentence.split()[0])
        encoded: List = self.encode_word_by_word(sentence)
        return [special_tokens[0]] + [tokens[0] for tokens in encoded] + [special_tokens[-1]]

    def encode_single_word(self, word: str) -> List:
        return [self.encoder.encode(word, add_special_tokens=False)[0]]
