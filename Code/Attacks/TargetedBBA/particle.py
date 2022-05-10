from functools import total_ordering
from typing import Optional

import numpy as np
from keras.models import Model

from Attacks.DistributedBBA.distribution_scheme import RoundRobinDistribution, DistanceBasedDistributionScheme
from Attacks.DistributedBBA.node import Node
from Attacks.DistributedBBA.node_manager import NodeManager, L2NodeManager, EmbeddedNodeManager, \
    ResettingEmbeddedNodeManager, InsertResettingEmbeddedNodeManager
from Attacks.TargetedBBA.utils import line_search_to_boundary


@total_ordering
class Particle:

    def __init__(self, i: int, init: np.ndarray = None, target_img: np.ndarray = None, target_label: int = 0,
                 model: Model = None, swarm=None, is_targeted: bool = True, source_step: float = 1e-2,
                 spherical_step: float = 5e-2, steps_per_iteration: int = 50, source_step_multiplier_up: float = 1.05,
                 source_step_multiplier_down: float = 0.6, use_node_manager: bool = False, dataset=None,
                 distance_based: Optional[str] = None, history_len=10, threshold: float = 0.02):
        self.id: int = i
        self.position: np.ndarray = init
        self.velocity: np.ndarray = np.random.randn(*self.position.shape) - 0.5
        self.position, calls = line_search_to_boundary(model, target_img, self.position, target_label, True, True)
        self.current_label: int = -1
        self.is_adversarial: bool = True
        self.is_targeted: bool = is_targeted

        self.target_image: np.ndarray = target_img
        self.target_label: int = target_label

        self.model: Model = model
        self.swarm = swarm
        self.swarm.total_queries += calls

        self.use_node_manager = use_node_manager
        self.distance_based = distance_based
        if self.use_node_manager:
            if distance_based is None:
                self.node_manager = NodeManager(dataset=dataset, nodes=swarm.nodes)
            elif distance_based == 'l2':
                self.node_manager = L2NodeManager(dataset=dataset, nodes=swarm.nodes, history_len=history_len)
            elif distance_based == 'embedded':
                self.node_manager = EmbeddedNodeManager(dataset=dataset, nodes=swarm.nodes, history_len=history_len)
            elif distance_based == 'resetting_embedded':
                self.node_manager = ResettingEmbeddedNodeManager(dataset=dataset, nodes=swarm.nodes,
                                                                 history_len=history_len)
            elif distance_based == 'insert_resetting_embedded':
                self.node_manager = InsertResettingEmbeddedNodeManager(dataset=dataset, nodes=swarm.nodes,
                                                                       threshold=threshold,
                                                                       history_len=history_len)
            else:
                raise ValueError

        self.steps_per_iteration: int = steps_per_iteration
        self.fitness: float = np.infty
        self.calculate_fitness()
        self.best_position: np.ndarray = self.position
        self.best_fitness: float = self.fitness

        self.source_step: float = source_step  # 0.25 for MNIST
        self.source_step_multiplier_up = source_step_multiplier_up
        self.source_step_multiplier_down = source_step_multiplier_down
        self.spherical_step: float = spherical_step
        self.maximum_diff: float = 0.4
        self.c1, self.c2 = self.select_cs()
        self.has_been_decreased = False
        self.non_adversarial_count = 0

    def __eq__(self, other: 'Particle') -> bool:
        return self.fitness == other.fitness and self.best_fitness == other.best_fitness

    def __lt__(self, other: 'Particle') -> bool:
        return self.fitness < other.fitness or (
                self.fitness == other.fitness and self.best_fitness < other.best_fitness)

    def get_node(self) -> Optional[Node]:
        if not self.use_node_manager:
            if self.swarm.distributed_attack is not None:
                if isinstance(self.swarm.distributed_attack.distribution_scheme, RoundRobinDistribution):
                    mapping = self.swarm.distribution_scheme.get_mapping()
                    return self.swarm.nodes[mapping[self.id]]
                elif isinstance(self.swarm.distributed_attack.distribution_scheme, DistanceBasedDistributionScheme):
                    mapping = self.swarm.distribution_scheme.get_mapping()
                    return self.swarm.nodes[mapping[self.id]]
                else:
                    return None
            else:
                return None
        else:
            return self.node_manager

    def calculate_fitness(self) -> None:
        prediction = np.argmax(self.model(np.expand_dims(self.position, axis=0)))
        self.swarm.total_queries += 1
        self.current_label = prediction

        node = self.get_node()
        if node is not None:
            node.add_to_detector(self.position)

        if (prediction == self.target_label) != self.is_targeted:
            # No longer adversarial
            self.fitness = np.infty
            self.is_adversarial = False
        else:
            self.fitness = np.linalg.norm(self.position - self.target_image)
            self.is_adversarial = True

    def update_position(self) -> None:
        if self.is_adversarial:
            self.has_been_decreased = False
            mask = np.abs(self.position - self.target_image)
            mask /= np.max(mask)  # scale to [0,1]
            mask = mask ** 0.5  # weaken the effect a bit.
            node = self.get_node()
            self.position = self.swarm.attack.run_attack(self.target_image, self.target_label, True, self.position,
                                                         (lambda: self.steps_per_iteration - self.swarm.attack.calls),
                                                         source_step=self.source_step,
                                                         spherical_step=self.spherical_step, mask=mask, pso=True,
                                                         node=node, dimensions=self.position.shape)
            self.swarm.total_queries += self.swarm.attack.calls
            self.source_step *= self.source_step_multiplier_up
        else:
            # print(f"Particle {self.id} is not longer adversarial!")
            self.update_velocity()
            self.position += self.velocity
            self.position = np.clip(self.position, 0, 1)
            if not self.has_been_decreased:
                self.source_step *= self.source_step_multiplier_down
                self.has_been_decreased = True
                self.non_adversarial_count += 1

    def update_bests(self) -> None:
        self.calculate_fitness()
        # print(f"Particle {self.id}: {self.fitness}")
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position

        if self.non_adversarial_count >= 20:
            self.source_step = 0.002

    def update_velocity(self, c1: float = 2., c2: float = 2.) -> None:
        w = self.calculate_w(1., 0., 1000)
        particle_best_delta = c2 * (self.best_position - self.position) * np.random.uniform(0., 1.,
                                                                                            self.target_image.shape)
        swarm_best_delta = c1 * (self.swarm.best_position - self.position) * np.random.uniform(0., 1.,
                                                                                               self.target_image.shape)
        deltas = particle_best_delta + swarm_best_delta
        self.velocity = w * np.clip(self.velocity + deltas, -1 * self.maximum_diff, self.maximum_diff)

    def calculate_w(self, w_start: float, w_end: float, max_queries: int) -> float:
        if self.swarm.n_particles == 1:
            return 1.

        if np.all(np.equal(self.best_position, self.swarm.best_position)):
            return w_end
        else:
            return w_end + ((w_start - w_end) * (1 - (self.swarm.iteration / max_queries)))

    def select_cs(self) -> (float, float):
        a1, a2 = 1., 2.
        if self.id % 2 == 0:
            c1, c2 = max(a1, a2), min(a1, a2)
        else:
            c1, c2 = min(a1, a2), max(a1, a2)
        return c1, c2
