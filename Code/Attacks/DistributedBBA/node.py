from collections import deque

import numpy as np

from Attacks.TargetedBBA.sampling_provider import create_perlin_noise
from Defense.detector import SimilarityDetector


class Node:
    def __init__(self, idx, dataset, k=50, chunk_size=1000, weights_path_mnist='../../Defense/MNISTencoder.h5',
                 flush_buffer_after_detection=True, insert_noise=None):
        self.idx = idx
        self.queries = deque(maxlen=chunk_size)
        self.particles = deque(maxlen=chunk_size)
        self.insert_noise = insert_noise
        self.node_calls = 0
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
        self.node_calls += 1
        if self.insert_noise is not None:
            if self.node_calls == self.insert_noise.insert_every:
                self.node_calls = 0
                if self.insert_noise.insert_noise is None:
                    random_indexes = np.random.choice(self.insert_noise.insert_from.shape[0],
                                                      size=self.insert_noise.insert_n)
                    for random_index in random_indexes:
                        random_query = self.insert_noise.insert_from[random_index]
                        self.detector.process(np.expand_dims(random_query, axis=0))
                else:
                    if self.insert_noise.insert_noise == 'uniform':
                        for _ in range(self.insert_noise.insert_n):
                            random_query = np.random.uniform(0, 1, size=self.insert_noise.insert_shape)
                            self.detector.process(np.expand_dims(random_query, axis=0))
                    elif self.insert_noise.insert_noise == 'perlin':
                        for _ in range(self.insert_noise.insert_n):
                            random_query = create_perlin_noise(np.array(self.insert_noise.insert_shape),
                                                               freq=np.random.randint(1, 40))
                            self.detector.process(np.expand_dims(random_query, axis=0))
                    elif self.insert_noise.insert_noise == 'mixed':
                        for _ in range(self.insert_noise.insert_n):
                            idx = np.random.randint(0, 2)
                            if idx == 0:
                                random_query = self.insert_noise.insert_from[
                                    np.random.randint(0, self.insert_noise.insert_from.shape[0])]
                            elif idx == 1:
                                random_query = np.random.uniform(0, 1, size=self.insert_noise.insert_shape)
                            elif idx == 2:
                                random_query = create_perlin_noise(np.array(self.insert_noise.insert_shape))
                            else:
                                raise ValueError
                            self.detector.process(np.expand_dims(random_query, axis=0))
                    else:
                        raise ValueError
