import operator
from dataclasses import dataclass, fields, astuple


@dataclass(init=True)
class HSJAPosition:
    init_n_evals: int = 0
    max_n_evals: int = 0
    step_size_decrease: float = 0.
    gamma: float = 0.

    @property
    def shape(self) -> tuple:
        return len(fields(self)),

    def __iter__(self):
        return iter(astuple(self))

    def __add__(self, other):
        return HSJAPosition(*(operator.add(*pair) for pair in zip(self, other)))

    def __sub__(self, other):
        return HSJAPosition(*(operator.sub(*pair) for pair in zip(self, other)))

    def __mul__(self, other):
        return HSJAPosition(*(operator.mul(*pair) for pair in zip(self, other)))
