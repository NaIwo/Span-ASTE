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
