from abc import ABC, abstractmethod
from collections import deque


class DistributionScheme(ABC):

    def __call__(self, mapping, **kwargs) -> None:
        return self.rotate(mapping, **kwargs)

    @abstractmethod
    def rotate(self, mapping: deque, **kwargs) -> None:
        pass


class FixedDistribution(DistributionScheme):

    def rotate(self, mapping: deque, **kwargs) -> None:
        return mapping.rotate(0)

    def __str__(self) -> str:
        return 'fixed'


class RoundRobinDistribution(DistributionScheme):

    def rotate(self, mapping: deque, r: int = 1) -> None:
        return mapping.rotate(r)

    def __str__(self) -> str:
        return 'round_robin'
