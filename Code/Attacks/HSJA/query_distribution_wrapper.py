import numpy as np

from Attacks.DistributedBBA.node import Node


class QueryDistributionWrapper:

    def __init__(self, n_nodes, flush_buffer_after_detection: bool = True):
        self.n_nodes = n_nodes
        self.nodes = [Node(i, 'mnist', weights_path_mnist='../../Defense/MNISTencoder.h5',
                           flush_buffer_after_detection=flush_buffer_after_detection) for i in range(self.n_nodes)]
        self.idx = 0
        self.n_queries = 0

    def predict(self, model, images):
        for image in images:
            self.nodes[self.idx].add_to_detector(image.reshape((28, 28, 1)))
            self.idx += 1
            self.idx %= self.n_nodes
        self.n_queries += images.shape[0]
        return model.predict(images.reshape((images.shape[0],) + (28, 28, 1)))

    def get_n_detections(self):
        detections = 0
        for node in self.nodes:
            detections += len(node.detector.get_detections())
        return detections
