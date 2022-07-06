from typing import TypeVar, Union

SC = TypeVar('SC', bound='StatsCounter')


class StatsCounter:
    def __init__(self, numerator: float = 0.0, *, denominator: float = 0.0):
        self.numerator: float = numerator
        self.denominator: float = denominator

    def __radd__(self, other: SC):
        return self.__add__(other)

    def __add__(self, other: SC):
        return StatsCounter(numerator=self.numerator + other.numerator,
                            denominator=self.denominator + other.denominator)

    def __repr__(self):
        return round(self.numerator / self.denominator, 4) if self.denominator else int(self.numerator)

    def __str__(self):
        return str(round(self.numerator / self.denominator, 4)) if self.denominator else str(int(self.numerator))

    def number(self) -> Union[int, float]:
        return round(self.numerator / self.denominator, 4) if self.denominator else int(self.numerator)
