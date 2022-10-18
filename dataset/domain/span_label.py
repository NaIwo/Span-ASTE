from .sentence import Sentence, Triplet
from .const import SpanCode

import numpy as np


def get_span_label_from_sentence(sentence: Sentence) -> np.ndarray:
    chunk: np.ndarray = np.full(shape=sentence.emb_sentence_length, fill_value=SpanCode.NOT_SPLIT)

    def fill_span(start_idx: int, end_idx: int):
        chunk[start_idx:end_idx] = SpanCode.SPLIT
        chunk[start_idx] = SpanCode.BEGIN_SPLIT

    triplet: Triplet
    for triplet in sentence.triplets:
        # Remember to add 1 in this place, not in fill_chunk func.
        fill_span(start_idx=sentence.get_index_after_encoding(triplet.aspect_span.start_idx),
                  end_idx=sentence.get_index_after_encoding(triplet.aspect_span.end_idx + 1))
        fill_span(start_idx=sentence.get_index_after_encoding(triplet.opinion_span.start_idx),
                  end_idx=sentence.get_index_after_encoding(triplet.opinion_span.end_idx + 1))

    chunk[chunk.shape[0] - sentence.encoder.offset:] = SpanCode.NOT_SPLIT
    chunk[:sentence.encoder.offset] = SpanCode.NOT_SPLIT
    return chunk
