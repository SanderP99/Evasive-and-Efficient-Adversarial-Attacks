from collections import deque
from typing import Optional, List

from keras.models import Model, load_model

from Attacks.DistributedBBA.distributed_bba import DistributedBiasedBoundaryAttack
from Attacks.DistributedBBA.experiment import Experiment
from Attacks.DistributedBBA.node import Node


class CombinedDistributedBiasedBoundaryAttack:
    def __init__(self, experiments: List[Experiment], n_particles: int, model: Model,
                 distribution_scheme, mapping: Optional[deque] = None,
                 n_nodes: Optional[int] = None, dataset=None, source_step: float = 1e-2,
                 spherical_step: float = 5e-2, steps_per_iteration: int = 50, source_step_multiplier_up: float = 1.05,
                 source_step_multiplier_down: float = 0.6, notify: bool = False):
        self.n_experiments = len(experiments)
        self.attacks = []
        self.nodes = [
            Node(i, dataset, weights_path_mnist=f'../../Defense/{str.upper(dataset)}encoder.h5', notify=notify) for i
            in range(n_nodes)]
        self.node_idx = 0
        self.n_nodes = n_nodes
        self.n_particles = n_particles
        self.total_queries = 0

        for experiment in experiments:
            experiment.set_random_inits(n_particles)
            self.attacks.append(DistributedBiasedBoundaryAttack(n_particles, experiment.random_inits, experiment.x_orig,
                                                                experiment.y_target,
                                                                model, distribution_scheme, mapping, n_nodes, dataset,
                                                                source_step,
                                                                spherical_step, steps_per_iteration,
                                                                source_step_multiplier_up, source_step_multiplier_down,
                                                                True, 'combination', notify=notify))

    def attack(self):
        for a in self.attacks:
            a.attack()

        # Submit the queries
        self.submit_queries()

        # Clear the buffers
        for a in self.attacks:
            for p in a.swarm.particles:
                p.node_manager.clear()

        self.total_queries = sum([a.swarm.total_queries for a in self.attacks])

    def submit_queries(self):
        for i in range(self.n_particles):
            for a in self.attacks:
                p = a.swarm.particles[i]
                for q in p.node_manager.queries:
                    self.nodes[self.node_idx].add_to_detector(q)
                    self.node_idx += 1
                    self.node_idx %= self.n_nodes
