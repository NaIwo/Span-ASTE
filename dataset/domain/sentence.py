from ASTE.dataset.encoders.encoder import encoder

from ast import literal_eval
from typing import List, Tuple, Dict, TypeVar

S = TypeVar('S', bound='Span')
T = TypeVar('T', bound='Triplet')


class Sentence:
    SEP: str = '#### #### ####'

    def __init__(self, raw_sentence: str):
        splitted_sentence: List = raw_sentence.split(Sentence.SEP)
        self.sentence: str = splitted_sentence[0]
        triplets_info: List[Tuple] = literal_eval(splitted_sentence[1])

        self.encoded_sentence: List = encoder.encode(sentence=self.sentence)
        self.encoded_words_in_sentence: List = encoder.encode_word_by_word(sentence=self.sentence)
        self.sub_words_lengths: List[int] = [len(word) - 1 for word in self.encoded_words_in_sentence]
        self.sentence_length: int = len(self.sentence.split())
        self.encoded_sentence_length: int = len(self.encoded_sentence)

        self.triplets: List[Triplet] = list()
        for triplet_info in triplets_info:
            self.triplets.append(Triplet.from_triplet_info(triplet_info, self.sentence))

    def get_index_after_encoding(self, idx: int) -> int:
        return encoder.offset + idx + sum(self.sub_words_lengths[:idx])


class Span:
    def __init__(self, start_idx: int, end_idx: int, words: List[str]):
        self.start_idx: int = start_idx
        self.end_idx: int = end_idx
        self.span_words: List[str] = words

    @classmethod
    def from_range(cls, span_range: List[int], sentence: str) -> S:
        if len(span_range) == 1:
            span_range.append(span_range[0])
        words: List[str] = sentence.split()[span_range[0]:span_range[1] + 1]
        return Span(start_idx=span_range[0], end_idx=span_range[1], words=words)

    def __str__(self) -> str:
        return str({
            'start idx': self.start_idx,
            'end idx': self.end_idx,
            'span words': self.span_words
        })

    def __repr__(self):
        return self.__str__()


class Triplet:
    def __init__(self, aspect_span: Span, opinion_span: Span, sentiment: str):
        self.aspect_span: Span = aspect_span
        self.opinion_span: Span = opinion_span
        self._sentiment: str = sentiment
        self._mapper: Dict = {
            'POS': 1,
            'NEU': 0,
            'NEG': -1
        }

    @property
    def sentiment(self) -> int:
        return self._mapper[self._sentiment]

    @classmethod
    def from_triplet_info(cls, triplet_info: Tuple, sentence: str) -> T:
        return Triplet(
            aspect_span=Span.from_range(triplet_info[0], sentence),
            opinion_span=Span.from_range(triplet_info[1], sentence),
            sentiment=triplet_info[2]
        )

    def __str__(self) -> str:
        return str({
            'aspect span': self.aspect_span,
            'opinion span': self.opinion_span,
            'sentiment': self._sentiment
        })

    def __repr__(self):
        return self.__str__()
