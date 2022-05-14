from typing import List

import numpy as np
from keras.models import load_model

from Attacks.DistributedBBA.node import Node
from MNIST.setup_cifar import CIFAR
from MNIST.setup_mnist import MNIST


class NodeManager(Node):
    def __init__(self, nodes: List[Node], dataset: str):
        super().__init__(-1, dataset)
        self.nodes = nodes
        self.n_nodes = len(nodes)
        self.idx = 0

    def add_to_detector(self, query: np.ndarray) -> None:
        self.nodes[self.idx].add_to_detector(query)
        self.idx += 1
        self.idx %= self.n_nodes


class L2NodeManager(NodeManager):
    def __init__(self, nodes: List[Node], dataset: str, history_len=10):
        super().__init__(nodes, dataset)
        self.history_len = history_len

        if dataset == 'mnist':
            shape = (28, 28, 1)
        elif dataset == 'cifar':
            shape = (32, 32, 3)
        else:
            raise ValueError

        self.history: np.ndarray = np.zeros((self.n_nodes, self.history_len) + shape)

    def add_to_detector(self, query: np.ndarray) -> None:
        distances = self.calculate_distances(query)
        best_idx = np.argmax(distances)
        self.nodes[best_idx].add_to_detector(query)
        self.history[best_idx][self.idx % self.history_len] = query
        self.idx += 1

    def calculate_distances(self, query):
        distances = np.zeros(self.n_nodes)
        for node_idx, node in enumerate(self.history):
            for historic_position in node:
                if np.any(historic_position):
                    distances[node_idx] += np.linalg.norm(historic_position - query)
                else:
                    distances[node_idx] += 1
        return distances


class EmbeddedNodeManager(NodeManager):
    def __init__(self, nodes: List[Node], dataset: str, history_len=10):
        super().__init__(nodes, dataset)
        self.history_len = history_len
        self.history = np.zeros((self.n_nodes, self.history_len) + (128,))

        if dataset == 'mnist':
            self.encoder = load_model('../../Defense/MNISTAttackencoder.h5', compile=False)
        elif dataset == 'cifar':
            self.encoder = load_model('../../Defense/CIFARAttackencoder.h5', compile=False)
        else:
            raise ValueError

    def add_to_detector(self, query: np.ndarray) -> None:
        embedded_query = self.encoder(np.expand_dims(query, axis=0))
        distances = self.calculate_distances(embedded_query)
        best_idx = np.argmax(distances)
        self.nodes[best_idx].add_to_detector(query)
        self.history[best_idx][self.idx % self.history_len] = embedded_query
        self.idx += 1

    def calculate_distances(self, query):
        distances = np.zeros(self.n_nodes)
        for node_idx, node in enumerate(self.history):
            for historic_position in node:
                if np.any(historic_position):
                    distances[node_idx] += np.linalg.norm(historic_position - query)
                else:
                    distances[node_idx] += 1
        return distances


class ResettingEmbeddedNodeManager(EmbeddedNodeManager):
    def add_to_detector(self, query: np.ndarray) -> None:
        embedded_query = self.encoder(np.expand_dims(query, axis=0))
        distances = self.calculate_distances(embedded_query)
        best_idx = np.argmax(distances)
        is_attack = self.nodes[best_idx].add_to_detector(query)
        if is_attack:
            self.history[best_idx] = 0  # Attack flagged so remove buffer
        else:
            self.history[best_idx][self.idx % self.history_len] = embedded_query
        self.idx += 1

    def calculate_distances(self, query):
        distances = np.zeros(self.n_nodes)
        for node_idx, node in enumerate(self.history):
            for historic_position in node:
                if np.any(historic_position):
                    distances[node_idx] += np.linalg.norm(historic_position - query)
                else:
                    distances[node_idx] += 2
        return distances


class InsertResettingEmbeddedNodeManager(ResettingEmbeddedNodeManager):
    def __init__(self, nodes: List[Node], dataset: str, threshold: float, history_len=10):
        super().__init__(nodes, dataset, history_len)
        self.threshold = threshold
        if dataset == 'mnist':
            self.training_data = MNIST().train_data
        elif dataset == 'cifar':
            self.training_data = CIFAR().train_data
        else:
            raise ValueError

    def add_to_detector(self, query: np.ndarray) -> None:
        embedded_query = self.encoder(np.expand_dims(query, axis=0))
        distances = self.calculate_distances(embedded_query)
        best_idx = np.argmax(distances)
        best_distance = distances[best_idx]
        if best_distance < self.threshold:
            # Very close to detection
            self.add_noise_to_detector(best_idx)
        is_attack = self.nodes[best_idx].add_to_detector(query)
        if is_attack:
            self.history[best_idx] = 0  # Attack flagged so remove buffer
        else:
            self.history[best_idx][self.idx % self.history_len] = embedded_query
        self.idx += 1

    def add_noise_to_detector(self, node_idx) -> None:
        random_query = self.training_data[np.random.randint(0, self.training_data.shape[0])]
        embedded_query = self.encoder(np.expand_dims(random_query, axis=0))
        is_attack = self.nodes[node_idx].add_to_detector(random_query)
        if is_attack:
            self.history[node_idx] = 0  # Attack flagged so remove buffer
        else:
            self.history[node_idx][self.idx % self.history_len] = embedded_query
        self.idx += 1


class InsertEmbeddedNodeManager(InsertResettingEmbeddedNodeManager):
    def add_to_detector(self, query: np.ndarray) -> None:
        embedded_query = self.encoder(np.expand_dims(query, axis=0))
        distances = self.calculate_distances(embedded_query)
        best_idx = np.argmax(distances)
        best_distance = distances[best_idx]
        if best_distance < self.threshold:
            # Very close to detection
            self.add_noise_to_detector(best_idx)
        _ = self.nodes[best_idx].add_to_detector(query)
        self.history[best_idx][self.idx % self.history_len] = embedded_query
        self.idx += 1

    def add_noise_to_detector(self, node_idx) -> None:
        random_query = self.training_data[np.random.randint(0, self.training_data.shape[0])]
        embedded_query = self.encoder(np.expand_dims(random_query, axis=0))
        _ = self.nodes[node_idx].add_to_detector(random_query)
        self.history[node_idx][self.idx % self.history_len] = embedded_query
        self.idx += 1
