import numpy as np
import sklearn.metrics.pairwise as pairwise
import gc
from Defense.vaencoder import mnist_encoder


def calculate_thresholds(training_data, k, encoder=lambda x: x, p=1000, up_to_k=False):
    data = np.array(encoder(training_data))[0]
    print(data.shape)

    dists = None
    for i in range(data.shape[0] // p):
        print(i)
        gc.collect()
        distance_mat = pairwise.pairwise_distances(data[i * p:(i + 1) * p, :], Y=data)
        distance_mat = np.sort(distance_mat, axis=-1)
        distance_mat_k = distance_mat[:, :k]
        if dists is None:
            dists = distance_mat_k
        else:
            dists = np.concatenate([dists, distance_mat_k])

    # distance_mat = np.concatenate(dists)
    start = 0 if up_to_k else k

    thresholds = []
    ks = []
    for i in range(start, k + 1):
        print(i)
        dist_to_k_neighbors = distance_mat[:, :i + 1]
        avg_dist = dist_to_k_neighbors.mean(axis=-1)
        threshold = np.percentile(avg_dist, 0.1)
        ks.append(i)
        thresholds.append(threshold)
    return ks, thresholds


class Detector:

    def __init__(self, k, threshold=None, training_data=None, chunk_size=1000, weights_path='./encoder_1.h5'):
        self.k = k
        self.threshold = threshold
        self.training_data = training_data

        if self.threshold is None and self.training_data is None:
            raise ValueError("Must provide explicit detection threshold or training data to calculate threshold!")

        self._init_encoder(weights_path)

        if self.training_data is not None:
            print("Explicit threshold not provided...calculating threshold for K = %d" % k)
            _, self.thresholds = calculate_thresholds(self.training_data, self.k, self.encode, up_to_k=False)
            self.threshold = self.thresholds[-1]
            print("K = %d; set threshold to: %f" % (self.k, self.threshold))

        self.num_queries = 0
        self.buffer = []
        # self.memory = []
        self.chunk_size = chunk_size

        self.history = []  # Tracks number of queries (t) when attack was detected
        self.history_by_attack = []
        self.detected_dists = []  # Tracks knn-dist that was detected

    def _init_encoder(self, weights_path):
        self.encode = lambda x: x
        raise NotImplementedError("Must implement your own encode function!")

    def process(self, queries):
        queries = self.encode(queries)[0]
        for query in queries:
            self.process_query(query)

    def process_query(self, query):
        if len(self.buffer) < self.k:
            self.buffer.append(query)
            self.num_queries += 1
            return False

        all_dists = []
        if len(self.buffer) > 0:
            queries = np.stack(self.buffer, axis=0)
            dists = np.linalg.norm(queries - query, axis=-1)
            all_dists.append(dists)

        # for queries in self.memory:
        #     dists = np.linalg.norm(queries - query, axis=-1)
        #     all_dists.append(dists)

        dists = np.concatenate(all_dists)
        k_nearest_dists = np.partition(dists, self.k - 1)[:self.k, None]
        k_avg_dist = np.mean(k_nearest_dists)

        self.buffer.append(query)
        self.num_queries += 1

        if len(self.buffer) >= self.chunk_size:
            # self.memory.append(np.stack(self.buffer, axis=0))
            self.buffer = []

        is_attack = k_avg_dist < self.threshold
        if is_attack:
            self.history.append(self.num_queries)
            self.detected_dists.append(k_avg_dist)
            self.clear_memory()

    def clear_memory(self):
        self.buffer = []
        # self.memory = []

    def get_detections(self):
        epochs = []
        for i in range(len(self.history) - 1):
            epochs.append(self.history[i + 1] - self.history[i])
        return epochs


class SimilarityDetector(Detector):

    def _init_encoder(self, weights_path):
        self.encoder = mnist_encoder(weights=weights_path)
        self.encode = lambda x: self.encoder.predict(x)
