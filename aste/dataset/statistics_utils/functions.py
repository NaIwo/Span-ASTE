from ..domain.const import ASTELabels
from ..domain import Sentence, Triplet, Span
from .counter import StatsCounter

import numpy as np
from functools import lru_cache
from typing import Set, List, Callable, Optional, Tuple
from itertools import chain, permutations, product


# ============ Dataset statistics ============ #


def sentences_count(_) -> StatsCounter:
    return StatsCounter(numerator=1)


def sentence_mean_length(sentence: Sentence) -> StatsCounter:
    return StatsCounter(numerator=sentence.sentence_length, denominator=1.0)


def specific_span_count(sentence: Sentence, span_source: str) -> StatsCounter:
    uniques: Set = _get_words_from_unique_spans(sentence, span_source)
    return StatsCounter(numerator=len(uniques))


def triplets_count(sentence: Sentence) -> StatsCounter:
    return StatsCounter(numerator=len(sentence.triplets))


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
    spans: List = _get_words_from_span(sentence, span_source)
    _, count = np.unique(spans, return_counts=True)
    return StatsCounter(numerator=int(np.sum(count > 1)))


def specific_span_word_length_count(sentence: Sentence, span_source: str, func: Callable) -> StatsCounter:
    uniques: Set = _get_words_from_unique_spans(sentence, span_source)
    return StatsCounter(numerator=sum(map(lambda el: func(el.split()), uniques)))


def triplet_word_length_count(sentence: Sentence, aspect_func: Callable, opinion_func: Callable,
                              op: str = 'and') -> StatsCounter:
    aspect_spans: List = _get_words_from_span(sentence, 'aspect_span')
    opinion_spans: List = _get_words_from_span(sentence, 'opinion_span')
    aspect_counts: List = list(map(lambda el: aspect_func(el.split()), aspect_spans))
    opinion_counts: List = list(map(lambda el: opinion_func(el.split()), opinion_spans))
    all_counts: np.ndarray = np.array([aspect_counts, opinion_counts])
    if not all_counts.any():
        return StatsCounter(numerator=0)
    if op == 'and':
        return StatsCounter(numerator=np.where(all_counts[0] & all_counts[1])[0].shape[0])
    elif op == 'or':
        return StatsCounter(numerator=np.where(all_counts[0] | all_counts[1])[0].shape[0])


def specific_phrase_count(sentence: Sentence, phrase: str, span_source: Optional[str] = None,
                          do_lower_case: bool = True) -> StatsCounter:
    if span_source is not None:
        iterable: List = list(map(lambda el: el.split(), _get_words_from_unique_spans(sentence, span_source)))
    else:
        iterable: List = [sentence.sentence.split()]
    process: Callable = lambda word: word.lower() if do_lower_case else word
    return StatsCounter(numerator=sum([process(phrase) == process(word) for word in chain(*iterable)]))


# ============ Result statistics ============ #

def bad_pairing_count(true_sentence: Sentence, pred_sentence: Sentence) -> StatsCounter:
    ordered_pred_triplets: List = sorted(pred_sentence.triplets)
    ordered_true_triplets: List = sorted(true_sentence.triplets)

    not_intersected: List = _get_not_intersected_predictions(ordered_true_triplets, ordered_pred_triplets)

    result_count: int = 0

    new_triplets: List[Triplet] = _new_triplets_from_span_permutations(not_intersected)
    new_not_intersected: List[Triplet] = _get_not_intersected_predictions(ordered_true_triplets, new_triplets)

    result_count += (len(new_triplets) - len(new_not_intersected))

    return StatsCounter(numerator=result_count)


def bad_sentiment_count(true_sentence: Sentence, pred_sentence: Sentence) -> StatsCounter:
    sc = StatsCounter()
    true: Triplet
    pred: Triplet
    for true, pred in product(sorted(true_sentence.triplets), sorted(pred_sentence.triplets)):
        if true == pred:
            continue
        else:
            if (true.aspect_span == pred.aspect_span) and (true.opinion_span == pred.opinion_span):
                sc.numerator += (true.sentiment != pred.sentiment)

    return sc


def opposite_sentiment_count(true_sentence: Sentence, pred_sentence: Sentence) -> StatsCounter:
    sc = StatsCounter()
    true: Triplet
    pred: Triplet
    for true, pred in product(sorted(true_sentence.triplets), sorted(pred_sentence.triplets)):
        if true == pred:
            continue
        else:
            if (true.aspect_span == pred.aspect_span) and (true.opinion_span == pred.opinion_span):
                true_sentiment: ASTELabels = ASTELabels[true.sentiment]
                pred_sentiment: ASTELabels = ASTELabels[pred.sentiment]
                sc.numerator += (true_sentiment == ASTELabels.POS and pred_sentiment == ASTELabels.NEG) or (
                        true_sentiment == ASTELabels.NEG and pred_sentiment == ASTELabels.POS)

    return sc


def not_included_words(true_sentence: Sentence, pred_sentence: Sentence) -> List:
    results: List = list()
    for true, pred in _yield_not_equals_triplets(true_sentence, pred_sentence):
        results += _get_words_diff(true, pred)

    return results


def over_included_words(true_sentence: Sentence, pred_sentence: Sentence) -> List:
    results: List = list()
    for true, pred in _yield_not_equals_triplets(true_sentence, pred_sentence):
        results += _get_words_diff(pred, true)

    return results


# ============ Helpers ============ #

def _yield_not_equals_triplets(true_sentence: Sentence, pred_sentence: Sentence) -> Tuple[Sentence, Sentence]:
    true: Triplet
    pred: Triplet
    for true, pred in product(sorted(true_sentence.triplets), sorted(pred_sentence.triplets)):
        if true != pred:
            yield true, pred


def _get_words_diff(true: Triplet, pred: Triplet) -> List:
    results: List = list()
    aspect_intersect: Span = true.aspect_span.intersect(pred.aspect_span)
    if aspect_intersect:
        aspect_words: List = list(set(true.aspect_span.span_words) - set(aspect_intersect.span_words))
        results += aspect_words
    opinion_intersect: Span = true.opinion_span.intersect(pred.opinion_span)
    if opinion_intersect:
        opinion_words: List = list(set(true.opinion_span.span_words) - set(opinion_intersect.span_words))
        results += opinion_words
    return results


def _new_triplets_from_span_permutations(not_intersected: List[Triplet]) -> List[Triplet]:
    new_triplets: List = list()
    aspect: Triplet
    opinion: Triplet
    for aspect, opinion in permutations(not_intersected, r=2):
        new_triplets.append(Triplet(aspect_span=aspect.aspect_span,
                                    opinion_span=opinion.opinion_span,
                                    sentiment='NOT_RELEVANT'))
    return new_triplets


def _get_not_intersected_predictions(true: List[Triplet], pred: List[Triplet]) -> List[Triplet]:
    not_intersected: List = list()
    pred_single: Triplet
    true_single: Triplet
    intersect: bool
    for pred_single in pred:
        intersect = False
        for true_single in true:
            if true_single.intersect(pred_single):
                intersect = True
                break
        if not intersect:
            not_intersected.append(pred_single)
    return not_intersected


def _unique_span_triplet_generator(sentence: Sentence, span_source: str) -> [Triplet]:
    assert span_source.lower() in ('opinion_span', 'aspect_span'), f'Wrong source of span: {span_source}.'

    uniques: Set = _get_words_from_unique_spans(sentence, span_source).copy()
    triplet: Triplet
    for triplet in sentence.triplets:
        span_words: str = _join_spans(getattr(triplet, span_source).span_words)
        if span_words not in uniques:
            continue
        uniques.remove(span_words)
        yield triplet


@lru_cache(maxsize=None)
def _get_words_from_unique_spans(sentence: Sentence, span_source: str) -> Set:
    return set(_get_words_from_span(sentence, span_source))


@lru_cache(maxsize=None)
def _get_words_from_span(sentence: Sentence, span_source: str) -> List:
    assert span_source.lower() in ('opinion_span', 'aspect_span'), f'Wrong source od span: {span_source}.'
    spans: List = list()
    triplet: Triplet
    for triplet in sentence.triplets:
        spans.append(_join_spans(getattr(triplet, span_source).span_words))
    return spans


def _join_spans(span: List) -> str:
    return ' '.join(span)
