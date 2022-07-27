from .sentence import Sentence, Triplet
from .const import ChunkCode

import numpy as np


def get_chunk_label_from_sentence(sentence: Sentence) -> np.ndarray:
    chunk: np.ndarray = np.full(shape=sentence.encoded_sentence_length, fill_value=ChunkCode.NOT_SPLIT)

    def fill_chunk(start_idx: int, end_idx: int):
        chunk[start_idx:end_idx] = ChunkCode.SPLIT
        chunk[start_idx] = ChunkCode.BEGIN_SPLIT

    triplet: Triplet
    for triplet in sentence.triplets:
        # Remember to add 1 in this place, not in fill_chunk func.
        fill_chunk(start_idx=sentence.get_index_after_encoding(triplet.aspect_span.start_idx),
                   end_idx=sentence.get_index_after_encoding(triplet.aspect_span.end_idx + 1))
        fill_chunk(start_idx=sentence.get_index_after_encoding(triplet.opinion_span.start_idx),
                   end_idx=sentence.get_index_after_encoding(triplet.opinion_span.end_idx + 1))

    chunk[chunk.shape[0] - sentence.encoder.offset:] = ChunkCode.NOT_SPLIT
    chunk[:sentence.encoder.offset] = ChunkCode.NOT_SPLIT
    return chunk
