from abc import ABC, abstractmethod
from collections import deque


class DistributionScheme(ABC):

    def __call__(self, mapping, **kwargs):
        return self.rotate(mapping, **kwargs)

    @abstractmethod
    def rotate(self, mapping: deque, **kwargs) -> deque:
        pass


class FixedDistribution(DistributionScheme):

    def rotate(self, mapping: deque, **kwargs) -> deque:
        return mapping

    def __str__(self):
        return 'fixed'


class RoundRobinDistribution(DistributionScheme):

    def rotate(self, mapping, r=1):
        return mapping.rotate(r)

    def __str__(self):
        return 'round_robin'
