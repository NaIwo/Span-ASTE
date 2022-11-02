from typing import List, Union

from transformers import DebertaTokenizer, AutoTokenizer

from .base_encoder import BaseEncoder


class TransformerEncoder(BaseEncoder):
    def __init__(self):
        super().__init__(encoder_name='transformer tokenizer')
        self.encoder: Union[DebertaTokenizer, AutoTokenizer] = self.get_transformer_encoder_from_config()
        self.offset: int = 1

    def encode(self, sentence: str) -> List:
        return self.encoder.encode(sentence)

    def encode_single_word(self, word: str) -> List:
        return self.encoder.encode(word, add_special_tokens=False)


class TransformerEncoderWithoutSubwords(BaseEncoder):
    def __init__(self):
        super().__init__(encoder_name='transformer without subwords tokenizer')
        self.encoder: Union[DebertaTokenizer, AutoTokenizer] = self.get_transformer_encoder_from_config()
        self.offset: int = 1

    def encode(self, sentence: str) -> List:
        special_tokens: List = self.encoder.encode(sentence.split()[0])
        encoded: List = self.encode_word_by_word(sentence)
        return [special_tokens[0]] + [tokens[0] for tokens in encoded] + [special_tokens[-1]]

    def encode_single_word(self, word: str) -> List:
        return [self.encoder.encode(word, add_special_tokens=False)[0]]
