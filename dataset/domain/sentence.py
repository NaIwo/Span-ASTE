from ast import literal_eval
from typing import List, Tuple, TypeVar, Optional

from ASTE.dataset.encoders import BaseEncoder, BertEncoder

S = TypeVar('S', bound='Span')
T = TypeVar('T', bound='Triplet')


class Sentence:
    SEP: str = '#### #### ####'

    def __init__(self, raw_sentence: str,
                 encoder: BaseEncoder = BertEncoder(),
                 include_sub_words_info_in_mask: bool = True):
        self.encoder: BaseEncoder = encoder
        self.raw_line: str = raw_sentence
        splitted_sentence: List = raw_sentence.strip().split(Sentence.SEP)
        self.sentence: str = splitted_sentence[0]
        self.triplets: List[Triplet] = []
        # If data with labels
        if len(splitted_sentence) == 2:
            triplets_info: List[Tuple] = literal_eval(splitted_sentence[1])
            self.triplets = [Triplet.from_triplet_info(triplet_info, self.sentence) for triplet_info in triplets_info]

        self.encoded_sentence: List[int] = self.encoder.encode(sentence=self.sentence)
        self.encoded_words_in_sentence: List = self.encoder.encode_word_by_word(sentence=self.sentence)

        self.include_sub_words_info_in_mask: bool = include_sub_words_info_in_mask

        self._sub_words_lengths: List[int] = list()
        self._true_sub_words_lengths: List[int] = list()
        self._sub_words_mask: List[int] = list()
        self._true_sub_words_mask: List[int] = list()

        self._fill_sub_words_information()

        self.sentence_length: int = len(self.sentence.split())
        self.encoded_sentence_length: int = len(self.encoded_sentence)
        self.emb_sentence_length: int = len(self._sub_words_mask)

    def _fill_sub_words_information(self):
        word: List[int]
        for word in self.encoded_words_in_sentence:
            len_sub_word: int = len(word) - 1

            self._sub_words_lengths.append(len_sub_word * int(self.include_sub_words_info_in_mask))
            self._true_sub_words_lengths.append(len_sub_word)

            self._sub_words_mask += [1] + ([0] * (len_sub_word * int(self.include_sub_words_info_in_mask)))
            self._true_sub_words_mask += [1] + [0] * len_sub_word

        offset: List[int] = [0] * self.encoder.offset
        self._sub_words_mask = offset + self._sub_words_mask + offset
        self._true_sub_words_mask = offset + self._true_sub_words_mask + offset

    def get_sub_words_mask(self, force_true_mask: bool = False):
        return self._true_sub_words_mask if force_true_mask else self._sub_words_mask

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
        if idx < 0 or idx >= self.sentence_length:
            return -1
        return self.encoder.offset + idx + sum(self._sub_words_lengths[:idx])

    def get_index_before_encoding(self, idx: int) -> int:
        if idx < 0 or idx >= self.emb_sentence_length:
            return -1
        return sum(self._sub_words_mask[:idx])

    def agree_index(self, idx: int) -> int:
        return self.get_index_after_encoding(self.get_index_before_encoding(idx))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Sentence):
            raise NotImplemented

        return (self.triplets == other.triplets) and (self.sentence == other.sentence) and (
                self.encoded_sentence == other.encoded_sentence)

    def __hash__(self):
        return hash(self.raw_line)


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

    def __eq__(self, other) -> bool:
        return (self.start_idx == other.start_idx) and (self.end_idx == other.end_idx) and (
                self.span_words == other.span_words)

    def __lt__(self, other):
        if self.start_idx != other.start_idx:
            return self.start_idx < other.start_idx
        else:
            return self.end_idx < other.end_idx

    def __gt__(self, other):
        if self.start_idx != other.start_idx:
            return self.start_idx > other.start_idx
        else:
            return self.end_idx > other.end_idx

    def __bool__(self) -> bool:
        return (self.start_idx != -1) and (self.end_idx != -1) and (self.span_words != [])

    def intersect(self, other) -> S:
        start_idx: int = max(self.start_idx, other.start_idx)
        end_idx: int = min(self.end_idx, other.end_idx)
        if end_idx < start_idx:
            return Span(start_idx=-1, end_idx=-1, words=[])
        else:
            span_words: List = self._get_intersected_words(other, start_idx, end_idx)

            return Span(start_idx=start_idx, end_idx=end_idx, words=span_words)

    def _get_intersected_words(self, other, start_idx, end_idx) -> List:
        span_words: List
        if start_idx == self.start_idx:
            if end_idx == self.end_idx:
                span_words = self.span_words[:]
            else:
                span_words = self.span_words[:-(self.end_idx - end_idx)]
        else:
            if end_idx == other.end_idx:
                span_words = other.span_words[:]
            else:
                span_words = other.span_words[:-(other.end_idx - end_idx)]
        return span_words


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

    def __eq__(self, other) -> bool:
        return (self.aspect_span == other.aspect_span) and (self.opinion_span == self.opinion_span) and (
                self.sentiment == other.sentiment)

    def __lt__(self, other):
        if self.aspect_span != other.aspect_span:
            return self.aspect_span < other.aspect_span
        else:
            return self.opinion_span < other.opinion_span

    def __gt__(self, other):
        if self.aspect_span != other.aspect_span:
            return self.aspect_span > other.aspect_span
        else:
            return self.opinion_span > other.opinion_span

    def __bool__(self) -> bool:
        return bool(self.aspect_span) and bool(self.opinion_span)

    def intersect(self, other) -> Optional[T]:
        return Triplet(
            aspect_span=self.aspect_span.intersect(other.aspect_span),
            opinion_span=self.opinion_span.intersect(other.opinion_span),
            sentiment=None if self.sentiment != other.sentiment else self.sentiment
        )
