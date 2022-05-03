from enum import IntEnum


class ChunkCode(IntEnum):
    NOT_RELEVANT: int = -1
    SPLIT: int = 1
    NOT_SPLIT: int = 0


class ASTELabels(IntEnum):
    POS: int = 5
    NEU: int = 4
    NEG: int = 3
    OPINION: int = 2
    ASPECT: int = 1
    NOT_PAIR: int = 0
    NOT_RELEVANT: int = -1
