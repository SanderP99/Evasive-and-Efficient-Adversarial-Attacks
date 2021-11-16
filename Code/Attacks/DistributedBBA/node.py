from collections import deque

import numpy as np

from Defense.detector import SimilarityDetector


class Node:
    def __init__(self, idx, k=50, chunk_size=1000):
        self.idx = idx
        self.queries = deque(maxlen=chunk_size)
        self.particles = deque(maxlen=chunk_size)
        self.detector = SimilarityDetector(k=k, chunk_size=chunk_size, threshold=0.449137,
                                           weights_path='../../Defense/encoder_weights.h5')

    def add_query(self, query, particle_id):
        """
        FOR DEBUG ONLY
        """
        self.queries.append(query)
        self.particles.append(particle_id)

    def add_to_detector(self, query):
        self.detector.process(np.expand_dims(query, axis=0))
