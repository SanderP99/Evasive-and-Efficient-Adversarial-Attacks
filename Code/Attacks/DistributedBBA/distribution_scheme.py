from abc import ABC, abstractmethod
from collections import deque

import numpy as np
from keras.models import load_model


class DistributionScheme(ABC):

    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, **kwargs) -> None:
        return self.rotate(self.mapping, **kwargs)

    @abstractmethod
    def rotate(self, mapping: deque, **kwargs) -> None:
        pass

    def get_mapping(self, **kwargs) -> deque:
        return self.mapping


class FixedDistribution(DistributionScheme):

    def rotate(self, mapping: deque, **kwargs) -> None:
        self.mapping = mapping

    def __str__(self) -> str:
        return 'fixed'


class RoundRobinDistribution(DistributionScheme):

    def __init__(self, mapping, n_nodes=1, n_particles=1):
        super().__init__(mapping)
        self.n_nodes = n_nodes
        self.n_particles = n_particles
        self.full_mapping = None

        if n_nodes == n_particles:
            self.mapping = deque(range(self.n_particles))
            self.full_mapping = self.mapping
        elif n_nodes < n_particles:
            self.mapping = deque(range(self.n_nodes)) + deque(np.random.randint(0, n_nodes, n_particles - n_nodes))
            self.full_mapping = deque(range(self.n_nodes))
        else:
            self.full_mapping = deque(range(self.n_nodes))
            self.mapping = deque(list(self.full_mapping)[:n_particles])

    def rotate(self, mapping: deque, **kwargs) -> None:
        if 'r' in kwargs:
            r = kwargs['r']
        else:
            r = 1
        if self.n_nodes == self.n_particles:
            self.full_mapping.rotate(r)
            self.mapping = self.full_mapping
        elif self.n_nodes < self.n_particles:
            self.full_mapping.rotate(r)
            self.mapping = self.full_mapping + deque(
                np.random.randint(0, self.n_nodes, self.n_particles - self.n_nodes))
        else:
            self.full_mapping.rotate(r)
            self.mapping = deque(list(self.full_mapping)[:self.n_particles])

    def __str__(self) -> str:
        return 'round_robin'


class ModifiedRoundRobinDistribution(RoundRobinDistribution):
    def __str__(self):
        return 'modified_round_robin'


class DistanceBasedDistributionScheme(DistributionScheme):

    def __init__(self, mapping, n_nodes, history_len: int = 1, dataset: str = ''):
        if dataset == 'mnist':
            shape = (28, 28, 1)
        elif dataset == 'cifar':
            shape = (32, 32, 3)
        else:
            raise ValueError

        super().__init__(mapping)
        self.history_len: int = history_len
        self.n_nodes: int = n_nodes
        self.history: np.ndarray = np.zeros((self.n_nodes, self.history_len) + shape)
        self.idx: int = 0
        self.best_deque: deque = deque([])

    def rotate(self, mapping: deque, **kwargs) -> None:
        pass

    def __str__(self) -> str:
        return f'distance_based_{self.history_len}'

    def compute_distances(self, positions: np.ndarray) -> np.ndarray:
        distances = np.zeros((len(positions), self.n_nodes))
        for node_idx, node in enumerate(self.history):
            for historic_position in node:
                for idx, position in enumerate(positions):
                    distances[idx][node_idx] += np.linalg.norm(historic_position - position)
        return distances / self.n_nodes

    def compute_best_deque(self, distances: np.ndarray) -> deque:
        idxs = np.argmax(distances, axis=1)
        # assert len(idxs) == len(self.mapping)
        return deque(idxs)

    def add_positions_to_history(self, positions: np.ndarray, best_deque: deque) -> None:
        for position, node in zip(positions, best_deque):
            self.history[node][self.idx % self.history_len] = position


class EmbeddedDistanceBasedDistributionScheme(DistanceBasedDistributionScheme):
    def __init__(self, mapping, n_nodes, history_len: int = 1, dataset: str = ''):
        super().__init__(mapping, n_nodes, history_len, dataset)
        self.history = np.zeros((self.n_nodes, self.history_len) + (128,))
        if dataset == 'mnist':
            self.encoder = load_model('../../Defense/MNISTAttackencoder.h5', compile=False)
        elif dataset == 'cifar':
            self.encoder = load_model('../../Defense/CIFARAttackencoder.h5', compile=False)
        else:
            raise ValueError

    def __str__(self) -> str:
        return f'embedded_distance_based_{self.history_len}'

    def add_positions_to_history(self, positions: np.ndarray, best_deque: deque) -> None:
        for position, node in zip(positions, best_deque):
            embedded_position = self.encoder(position)
            self.history[node][self.idx % self.history_len] = embedded_position


class ResettingEmbeddedDistanceBasedDistributionScheme(EmbeddedDistanceBasedDistributionScheme):
    def __str__(self):
        return f'resetting_embedded_distance_based_{self.history_len}'


class InsertResettingEmbeddedDistanceBasedDistributionScheme(ResettingEmbeddedDistanceBasedDistributionScheme):
    def __str__(self):
        return f'insert_resetting_embedded_distance_based_{self.history_len}'


class InsertEmbeddedDistanceBasedDistributionScheme(ResettingEmbeddedDistanceBasedDistributionScheme):
    def __str__(self):
        return f'insert_embedded_distance_based_{self.history_len}'


class CombinationDistributionScheme(RoundRobinDistribution):
    def __init__(self, mapping, n_nodes=1, n_particles=1, n_experiments=1):
        super(CombinationDistributionScheme, self).__init__(mapping, n_nodes, n_particles)
        self.n_experiments = n_experiments

    def __str__(self):
        return f'combination_distribution_{self.n_experiments}'
