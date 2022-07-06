from .sentence import Sentence, Triplet
from .const import ChunkCode
from ASTE.utils import config

import numpy as np


def get_chunk_label_from_sentence(sentence: Sentence) -> np.ndarray:
    chunk: np.ndarray = np.full(shape=sentence.encoded_sentence_length, fill_value=_get_fill_value())

    def fill_chunk(start_idx: int, end_idx: int):
        chunk[start_idx:end_idx] = ChunkCode.NOT_SPLIT
        chunk[start_idx] = ChunkCode.SPLIT
        if end_idx != sentence.encoded_sentence_length - 1:
            chunk[end_idx] = ChunkCode.SPLIT

    triplet: Triplet
    for triplet in sentence.triplets:
        # Remember to add 1 in this place, not in fill_chunk func.
        fill_chunk(start_idx=sentence.get_index_after_encoding(triplet.aspect_span.start_idx),
                   end_idx=sentence.get_index_after_encoding(triplet.aspect_span.end_idx + 1))
        fill_chunk(start_idx=sentence.get_index_after_encoding(triplet.opinion_span.start_idx),
                   end_idx=sentence.get_index_after_encoding(triplet.opinion_span.end_idx + 1))

    chunk[chunk.shape[0] - sentence.encoder.offset:] = ChunkCode.NOT_RELEVANT
    chunk[:sentence.encoder.offset] = ChunkCode.NOT_RELEVANT
    return chunk


def _get_fill_value() -> int:
    return ChunkCode.NOT_RELEVANT if config['model']['chunker']['mode'].lower() == 'soft' else ChunkCode.NOT_SPLIT
