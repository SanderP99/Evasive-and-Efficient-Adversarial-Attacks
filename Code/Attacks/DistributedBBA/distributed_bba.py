from collections import deque
from typing import Optional

import numpy as np
from keras.models import Model

from Attacks.DistributedBBA.node import Node
from Attacks.TargetedBBA.bba_pso import ParticleBiasedBoundaryAttack


class DistributedBiasedBoundaryAttack:
    def __init__(self, n_particles: int, inits: np.ndarray, target_img: np.ndarray, target_label: int, model: Model,
                 distribution_scheme, mapping: Optional[deque] = None,
                 n_nodes: Optional[int] = None):
        if n_nodes is None:
            self.n_nodes: int = n_particles
        else:
            self.n_nodes: int = n_nodes

        self.nodes: list = [Node(i) for i in range(self.n_nodes)]
        self.mapping: deque = mapping
        self.distribution_scheme = distribution_scheme

        self.swarm: ParticleBiasedBoundaryAttack = ParticleBiasedBoundaryAttack(n_particles, inits, target_img,
                                                                                target_label, model, self)

    def attack(self) -> None:
        self.swarm.optimize()
        self.process_query()
        self.distribution_scheme(self.mapping)

    def add_to_nodes(self) -> None:
        """
        FOR DEBUG ONLY
        """
        for p, m in zip(self.swarm.particles, self.mapping):
            self.nodes[m].add_query(p.position, p.id)

    def process_query(self) -> None:
        for p, m in zip(self.swarm.particles, self.mapping):
            self.nodes[m].add_to_detector(p.position)
