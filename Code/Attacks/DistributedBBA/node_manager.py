from typing import List

import numpy as np

from Attacks.DistributedBBA.node import Node


class NodeManager(Node):
    def __init__(self, nodes: List[Node], dataset):
        super().__init__(-1, dataset)
        self.nodes = nodes
        self.n_nodes = len(nodes)
        self.idx = 0

    def add_to_detector(self, query: np.ndarray) -> None:
        self.nodes[self.idx].add_to_detector(query)
        self.idx += 1
        self.idx %= self.n_nodes
