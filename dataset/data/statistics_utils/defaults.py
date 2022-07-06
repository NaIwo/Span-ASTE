from ASTE.dataset.domain.const import ASTELabels

from typing import Dict
from functools import partial

from .functions import (
    sentences_count,
    sentence_mean_length,
    specific_span_count,
    triplets_count,
    word_mean_chunk_length,
    encoder_mean_chunk_length,
    specific_span_encoder_count,
    specific_span_word_count,
    specific_sentiment_count,
    one_to_many_count,
    specific_span_word_length_count,
    triplet_word_length_counts,
    specific_phrase_count
)

default_statistics_func: Dict = {
    'Number of sentences': sentences_count,
    'Mean sentence length': sentence_mean_length,
    'Number of opinion phrases': partial(specific_span_count, span_source='opinion_span'),
    'Number of aspect phrases': partial(specific_span_count, span_source='aspect_span'),
    'Number of triplets': triplets_count,
    'Mean length of chunk (with encoder)': encoder_mean_chunk_length,
    'Mean length of chunk (words)': word_mean_chunk_length,
    'Mean length of opinion phrases (with encoder)': partial(specific_span_encoder_count, span_source='opinion_span'),
    'Mean length of opinion phrases (words)': partial(specific_span_word_count, span_source='opinion_span'),
    'Mean length of aspect phrases (with encoder)': partial(specific_span_encoder_count, span_source='aspect_span'),
    'Mean length of aspect phrases (words)': partial(specific_span_word_count, span_source='aspect_span'),
    'Number of positive sentiment': partial(specific_sentiment_count, sentiment=ASTELabels.POS),
    'Number of neutral sentiment': partial(specific_sentiment_count, sentiment=ASTELabels.NEU),
    'Number of negative sentiment': partial(specific_sentiment_count, sentiment=ASTELabels.NEG),
    'Number of one-to-many relations (opinions)': partial(one_to_many_count, span_source='opinion_span'),
    'Number of one-to-many relations (aspects)': partial(one_to_many_count, span_source='aspect_span'),
    'Number of opinions with length = 1': partial(
        specific_span_word_length_count, span_source='opinion_span',
        func=lambda el: len(el) == 1),
    'Number of aspects with length = 1': partial(
        specific_span_word_length_count, span_source='aspect_span',
        func=lambda el: len(el) == 1),
    'Number of opinions with length > 1': partial(
        specific_span_word_length_count, span_source='opinion_span',
        func=lambda el: len(el) > 1),
    'Number of aspects with length > 1': partial(
        specific_span_word_length_count, span_source='aspect_span',
        func=lambda el: len(el) > 1),
    'Number of triplets where length of each span = 1': partial(
        triplet_word_length_counts,
        aspect_func=lambda el: len(el) == 1,
        opinion_func=lambda el: len(el) == 1),
    'Number of triplets where aspect span length > 1 and opinion span length = 1': partial(
        triplet_word_length_counts,
        aspect_func=lambda el: len(el) > 1,
        opinion_func=lambda el: len(el) == 1),
    'Number of triplets where opinion span length > 1 and aspect span length = 1': partial(
        triplet_word_length_counts,
        aspect_func=lambda el: len(el) == 1,
        opinion_func=lambda el: len(el) > 1),
    'Number of triplets where at least one span length > 1': partial(
        triplet_word_length_counts,
        aspect_func=lambda el: len(el) > 1,
        opinion_func=lambda el: len(el) > 1,
        op='or'),
}

default_phrases_func: Dict = {
    'Number of specific phrases': specific_phrase_count,
    'Number of specific phrases in aspect span': partial(specific_phrase_count, span_source='aspect_span'),
    'Number of specific phrases in opinion span': partial(specific_phrase_count, span_source='opinion_span'),
}
