from abc import ABC, abstractmethod
from collections import deque


class DistributionScheme(ABC):

    def __call__(self, mapping, **kwargs):
        return self.rotate(mapping, **kwargs)

    @abstractmethod
    def rotate(self, mapping: deque, **kwargs) -> deque:
        pass


class RoundRobinDistribution(DistributionScheme):

    def rotate(self, mapping, r=1):
        return mapping.rotate(r)
