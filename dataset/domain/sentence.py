from ASTE.dataset.encoders import BaseEncoder, BertEncoder

from ast import literal_eval
from typing import List, Tuple, TypeVar

S = TypeVar('S', bound='Span')
T = TypeVar('T', bound='Triplet')


class Sentence:
    SEP: str = '#### #### ####'

    def __init__(self, raw_sentence: str, encoder: BaseEncoder = BertEncoder()):
        self.encoder: BaseEncoder = encoder
        splitted_sentence: List = raw_sentence.split(Sentence.SEP)
        self.sentence: str = splitted_sentence[0]
        self.triplets: List[Triplet] = []
        # If data with labels
        if len(splitted_sentence) == 2:
            triplets_info: List[Tuple] = literal_eval(splitted_sentence[1])
            self.triplets = [Triplet.from_triplet_info(triplet_info, self.sentence) for triplet_info in triplets_info]

        self.encoded_sentence: List[int] = self.encoder.encode(sentence=self.sentence)
        self.encoded_words_in_sentence: List = self.encoder.encode_word_by_word(sentence=self.sentence)
        self.sub_words_lengths: List[int] = list()
        self.sub_words_mask: List[int] = list()
        word: List[int]
        for word in self.encoded_words_in_sentence:
            self.sub_words_lengths.append(len(word) - 1)
            self.sub_words_mask += [1] + [0] * (len(word) - 1)
        offset: List[int] = [0] * self.encoder.offset
        self.sub_words_mask = offset + self.sub_words_mask + offset

        self.sentence_length: int = len(self.sentence.split())
        self.encoded_sentence_length: int = len(self.encoded_sentence)

    def get_aspect_spans(self) -> List[Tuple[int, int]]:
        return self._get_selected_spans('aspect_span')

    def get_opinion_spans(self) -> List[Tuple[int, int]]:
        return self._get_selected_spans('opinion_span')

    def _get_selected_spans(self, span_source: str) -> List[Tuple[int, int]]:
        assert span_source in ('aspect_span', 'opinion_span'), f'Invalid span source: {span_source}!'
        spans: List = list()
        triplet: Triplet
        for triplet in self.triplets:
            # +1)-1 -> If end index is the same as start_idx and word is constructed from sub-tokens
            # end index is shifted by number equals to this sub-words count.
            span: Tuple = (self.get_index_after_encoding(getattr(triplet, span_source).start_idx),
                           self.get_index_after_encoding(getattr(triplet, span_source).end_idx + 1) - 1)
            spans.append(span)
        return spans

    def get_index_after_encoding(self, idx: int) -> int:
        return self.encoder.offset + idx + sum(self.sub_words_lengths[:idx])

    def get_index_before_encoding(self, idx: int) -> int:
        return idx - self.encoder.offset - sum(self.sub_words_lengths[:idx])


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
        self.sentiment: str = sentiment

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
            'sentiment': self.sentiment
        })

    def __repr__(self):
        return self.__str__()
