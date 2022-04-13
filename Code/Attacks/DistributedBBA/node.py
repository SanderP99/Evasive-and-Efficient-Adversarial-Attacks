from collections import deque

import numpy as np

from Defense.detector import SimilarityDetector


class Node:
    def __init__(self, idx, dataset, k=50, chunk_size=1000, weights_path_mnist='../../Defense/MNISTencoder.h5',
                 flush_buffer_after_detection=True):
        self.idx = idx
        self.queries = deque(maxlen=chunk_size)
        self.particles = deque(maxlen=chunk_size)
        if dataset == 'mnist':
            self.detector = SimilarityDetector(k=k, chunk_size=chunk_size, threshold=0.00926118174381554,
                                               # Threshold autoencoder: 0.449137
                                               weights_path=weights_path_mnist,
                                               clear_buffer_after_detection=flush_buffer_after_detection)
        elif dataset == 'cifar':
            self.detector = SimilarityDetector(k=k, chunk_size=chunk_size, threshold=0.021234,
                                               weights_path=weights_path_mnist,
                                               clear_buffer_after_detection=flush_buffer_after_detection)
        else:
            raise ValueError

    def add_query(self, query: np.ndarray, particle_id: int) -> None:
        """
        FOR DEBUG ONLY
        """
        self.queries.append(query)
        self.particles.append(particle_id)

    def add_to_detector(self, query: np.ndarray) -> None:
        self.detector.process(np.expand_dims(query, axis=0))
