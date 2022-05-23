import numpy as np

from Attacks.DistributedBBA.node import Node


class QueryDistributionWrapper:

    def __init__(self, n_nodes, flush_buffer_after_detection: bool = True, dataset='mnist'):
        self.n_nodes = n_nodes
        self.nodes = [Node(i, dataset=dataset, weights_path_mnist=f'../../Defense/{dataset.upper()}encoder.h5',
                           flush_buffer_after_detection=flush_buffer_after_detection) for i in range(self.n_nodes)]
        self.idx = 0
        self.n_queries = 0
        if dataset == 'mnist':
            self.shape = (28, 28, 1)
        elif dataset == 'cifar':
            self.shape = (32, 32, 3)
        else:
            raise ValueError

    def predict(self, model, images):
        for image in images:
            self.nodes[self.idx].add_to_detector(image.reshape(self.shape))
            self.idx += 1
            self.idx %= self.n_nodes
        self.n_queries += images.shape[0]
        return model.predict(images.reshape((images.shape[0],) + self.shape))

    def get_n_detections(self):
        detections = 0
        for node in self.nodes:
            detections += len(node.detector.get_detections())
        return detections
