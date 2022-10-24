from typing import List, Union

from aste.utils import config
from transformers import DebertaTokenizer, AutoTokenizer


class BaseEncoder:
    def __init__(self, encoder_name: str = 'basic tokenizer'):
        self.encoder_name: str = encoder_name
        self.offset: int = 0
        self.encoder = None

    def encode(self, sentence: str) -> List:
        return [len(word) for word in sentence.strip().split()]

    def encode_single_word(self, word: str) -> List:
        return [len(word)]

    def encode_word_by_word(self, sentence: str) -> List[List]:
        encoded_words: List = list()
        word: str
        for word in sentence.strip().split():
            encoded_words.append(self.encode_single_word(word))
            self.encoder.add_prefix_space = True
        self.encoder.add_prefix_space = False
        return encoded_words

    @staticmethod
    def get_transformer_encoder_from_config() -> Union[DebertaTokenizer, AutoTokenizer]:
        if 'deberta' in config['encoder']['transformer']['source']:
            return DebertaTokenizer.from_pretrained(config['encoder']['transformer']['source'])
        elif 'bert' in config['encoder']['transformer']['source']:
            return AutoTokenizer.from_pretrained(config['encoder']['transformer']['source'])
        else:
            raise Exception(f"We do not support this transformer model {config['encoder']['transformer']['source']}!")
