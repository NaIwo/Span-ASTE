from enum import IntEnum


class SpanCode(IntEnum):
    NOT_RELEVANT: int = -1
    BEGIN_SPLIT: int = 5
    BEGIN_OPINION: int = 4
    INSIDE_OPINION: int = 3
    BEGIN_ASPECT: int = 2
    INSIDE_ASPECT: int = 1
    NOT_SPLIT: int = 0


class ASTELabels(IntEnum):
    NEU: int = 5
    POS: int = 4
    NEG: int = 3
    OPINION: int = 2
    ASPECT: int = 1
    NOT_PAIR: int = 0
    NOT_RELEVANT: int = -1
