from ASTE.dataset.domain.const import ChunkCode, ASTELabels
from ASTE.dataset.domain.sentence import Sentence, Triplet
from ASTE.dataset.domain.chunk_label import get_chunk_label_from_sentence
from .counter import StatsCounter

import numpy as np
from itertools import chain
from functools import lru_cache
from typing import Set, List, Callable, Optional


def sentences_count(_) -> StatsCounter:
    return StatsCounter(numerator=1)


def sentence_mean_length(sentence: Sentence) -> StatsCounter:
    return StatsCounter(numerator=sentence.sentence_length, denominator=1.0)


def specific_span_count(sentence: Sentence, span_source: str) -> StatsCounter:
    uniques: Set = _get_unique_spans(sentence, span_source)
    return StatsCounter(numerator=len(uniques))


def triplets_count(sentence: Sentence) -> StatsCounter:
    return StatsCounter(numerator=len(sentence.triplets))


def encoder_mean_chunk_length(sentence: Sentence) -> StatsCounter:
    chunk_label: np.ndarray = get_chunk_label_from_sentence(sentence)
    return StatsCounter(numerator=len(sentence.sub_words_mask) - sentence.encoder.offset,
                        denominator=np.where(chunk_label == ChunkCode.SPLIT)[0].shape[0])


def word_mean_chunk_length(sentence: Sentence) -> StatsCounter:
    chunk_label: np.ndarray = get_chunk_label_from_sentence(sentence)
    return StatsCounter(numerator=sentence.sentence_length,
                        denominator=np.where(chunk_label == ChunkCode.SPLIT)[0].shape[0])


def specific_sentiment_count(sentence: Sentence, sentiment: ASTELabels) -> StatsCounter:
    sc = StatsCounter()

    triplet: Triplet
    for triplet in sentence.triplets:
        sc.numerator += (ASTELabels[triplet.sentiment] == sentiment)
    return sc


def specific_span_encoder_count(sentence: Sentence, span_source: str):
    sc = StatsCounter()
    triplet: [Triplet]
    for triplet in _unique_span_triplet_generator(sentence, span_source):
        start: int = sentence.get_index_after_encoding(getattr(triplet, span_source).start_idx)
        end: int = sentence.get_index_after_encoding(getattr(triplet, span_source).end_idx)
        length: int = end - start + 1
        sc.numerator += length
        sc.denominator += 1
    return sc


def specific_span_word_count(sentence: Sentence, span_source: str) -> StatsCounter:
    sc = StatsCounter()
    triplet: [Triplet]
    for triplet in _unique_span_triplet_generator(sentence, span_source):
        start: int = getattr(triplet, span_source).start_idx
        end: int = getattr(triplet, span_source).end_idx
        length: int = end - start + 1
        sc.numerator += length
        sc.denominator += 1
    return sc


def one_to_many_count(sentence: Sentence, span_source: str) -> StatsCounter:
    spans: List = _get_spans(sentence, span_source)
    _, count = np.unique(spans, return_counts=True)
    return StatsCounter(numerator=int(np.sum(count > 1)))


def specific_span_word_length_count(sentence: Sentence, span_source: str, func: Callable) -> StatsCounter:
    uniques: Set = _get_unique_spans(sentence, span_source)
    return StatsCounter(numerator=sum(map(lambda el: func(el.split()), uniques)))


def triplet_word_length_counts(sentence: Sentence, aspect_func: Callable, opinion_func: Callable,
                               op: str = 'and') -> StatsCounter:
    aspect_spans: List = _get_spans(sentence, 'aspect_span')
    opinion_spans: List = _get_spans(sentence, 'opinion_span')
    aspect_counts: List = list(map(lambda el: aspect_func(el.split()), aspect_spans))
    opinion_counts: List = list(map(lambda el: opinion_func(el.split()), opinion_spans))
    all_counts: np.ndarray = np.array([aspect_counts, opinion_counts])
    if op == 'and':
        return StatsCounter(numerator=np.where(all_counts[0] & all_counts[1])[0].shape[0])
    elif op == 'or':
        return StatsCounter(numerator=np.where(all_counts[0] | all_counts[1])[0].shape[0])


def specific_phrase_count(sentence: Sentence, phrase: str, span_source: Optional[str] = None,
                          do_lower_case: bool = True) -> StatsCounter:

    if span_source is not None:
        iterable: List = list(map(lambda el: el.split(), _get_unique_spans(sentence, span_source)))
    else:
        iterable: List = [sentence.sentence.split()]
    process: Callable = lambda word: word.lower() if do_lower_case else word
    return StatsCounter(numerator=sum([process(phrase) == process(word) for word in chain(*iterable)]))


def _unique_span_triplet_generator(sentence: Sentence, span_source: str) -> [Triplet]:
    assert span_source.lower() in ('opinion_span', 'aspect_span'), f'Wrong source of span: {span_source}.'

    uniques: Set = _get_unique_spans(sentence, span_source).copy()
    triplet: Triplet
    for triplet in sentence.triplets:
        span_words: str = _join_spans(getattr(triplet, span_source).span_words)
        if span_words not in uniques:
            continue
        uniques.remove(span_words)
        yield triplet


@lru_cache(maxsize=None)
def _get_unique_spans(sentence: Sentence, span_source: str) -> Set:
    return set(_get_spans(sentence, span_source))


def _get_spans(sentence: Sentence, span_source: str) -> List:
    assert span_source.lower() in ('opinion_span', 'aspect_span'), f'Wrong source od span: {span_source}.'
    spans: List = list()
    triplet: Triplet
    for triplet in sentence.triplets:
        spans.append(_join_spans(getattr(triplet, span_source).span_words))
    return spans


def _join_spans(span: List) -> str:
    return ' '.join(span)
