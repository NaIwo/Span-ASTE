from enum import IntEnum
from typing import Dict


class ChunkCode(IntEnum):
    NOT_RELEVANT: int = -1
    SPLIT: int = 1
    NOT_SPLIT: int = 0


SENTIMENT_MAPPER: Dict = {
    'POS': 1,
    'NEU': 0,
    'NEG': -1
}
