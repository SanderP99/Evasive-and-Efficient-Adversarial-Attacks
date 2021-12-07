from collections import deque

from Attacks.DistributedBBA.node import Node
from Attacks.TargetedBBA.bba_pso import ParticleBiasedBoundaryAttack


class DistributedBiasedBoundaryAttack:
    def __init__(self, n_particles, inits, target_img, target_label, model, distribution_scheme, mapping = None, n_nodes=None):
        if n_nodes is None:
            self.n_nodes = n_particles
        else:
            self.n_nodes = n_nodes

        self.nodes = [Node(i) for i in range(self.n_nodes)]
        self.mapping = mapping
        self.distribution_scheme = distribution_scheme

        self.swarm = ParticleBiasedBoundaryAttack(n_particles, inits, target_img, target_label, model, self)

    def attack(self):
        self.swarm.optimize()
        self.process_query()
        self.distribution_scheme(self.mapping)

    def add_to_nodes(self):
        """
        FOR DEBUG ONLY
        """
        for p, m in zip(self.swarm.particles, self.mapping):
            self.nodes[m].add_query(p.position, p.id)

    def process_query(self):
        for p, m in zip(self.swarm.particles, self.mapping):
            self.nodes[m].add_to_detector(p.position)
