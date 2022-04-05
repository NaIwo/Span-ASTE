from typing import List


class BaseEncoder:
    def __init__(self, encoder_name: str = 'basic tokenizer'):
        self.encoder_name: str = encoder_name
        self.offset: int = 0

    def encode(self, sentence: str) -> List:
        return [len(word) for word in sentence.strip().split()]

    def encode_single_word(self, word: str) -> List:
        return [len(word)]

    def encode_word_by_word(self, sentence: str) -> List[List]:
        encoded_words: List = list()
        word: str
        for word in sentence.strip().split():
            encoded_words.append(self.encode_single_word(word))

        return encoded_words
