from .sentence import Sentence, Triplet
from .const import ChunkCode

from typing import TypeVar
import numpy as np

L = TypeVar('L', bound='Label')


class Label:
    def __init__(self, chunk: np.ndarray):
        self.chunk: np.ndarray = chunk

    @classmethod
    def from_sentence(cls, sentence: Sentence) -> L:
        chunk: np.ndarray = np.full(shape=sentence.encoded_sentence_length, fill_value=int(ChunkCode.NOT_RELEVANT))

        def fill_chunk(start_idx: int, end_idx: int):
            chunk[start_idx:end_idx] = int(ChunkCode.NOT_SPLIT)
            chunk[start_idx] = int(ChunkCode.SPLIT)
            if end_idx != sentence.encoded_sentence_length - 1:
                chunk[end_idx] = int(ChunkCode.SPLIT)

        triplet: Triplet
        for triplet in sentence.triplets:
            # Remember to add 1 in this place, not in fill_chunk func.
            fill_chunk(start_idx=sentence.get_index_after_encoding(triplet.aspect_span.start_idx),
                       end_idx=sentence.get_index_after_encoding(triplet.aspect_span.end_idx+1))
            fill_chunk(start_idx=sentence.get_index_after_encoding(triplet.opinion_span.start_idx),
                       end_idx=sentence.get_index_after_encoding(triplet.opinion_span.end_idx+1))

        chunk[chunk.shape[0]-sentence.encoder.offset:] = int(ChunkCode.NOT_RELEVANT)
        chunk[:sentence.encoder.offset] = int(ChunkCode.NOT_RELEVANT)
        return cls(chunk=chunk)
