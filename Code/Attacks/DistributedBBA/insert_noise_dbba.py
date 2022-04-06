from collections import deque
from typing import Optional

import numpy as np
from keras.models import Model

from Attacks.DistributedBBA.distributed_bba import DistributedBiasedBoundaryAttack
from MNIST.setup_cifar import CIFAR
from MNIST.setup_mnist import MNIST
from Attacks.TargetedBBA.sampling_provider import create_perlin_noise


class InsertNoiseDistributedBiasedBoundaryAttack(DistributedBiasedBoundaryAttack):
    def __init__(self, n_particles: int, inits: np.ndarray, target_img: np.ndarray, target_label: int, model: Model,
                 distribution_scheme, mapping: Optional[deque] = None,
                 n_nodes: Optional[int] = None, dataset=None, source_step: float = 1e-2,
                 spherical_step: float = 5e-2, steps_per_iteration: int = 50, source_step_multiplier_up: float = 1.05,
                 source_step_multiplier_down: float = 0.6, insert_every: int = 50, insert_from: np.array = None,
                 insert_n: int = 1, insert_noise: Optional[str] = None):
        super().__init__(n_particles, inits, target_img, target_label, model, distribution_scheme, mapping=mapping,
                         n_nodes=n_nodes, dataset=dataset, source_step=source_step, spherical_step=spherical_step,
                         steps_per_iteration=steps_per_iteration, source_step_multiplier_up=source_step_multiplier_up,
                         source_step_multiplier_down=source_step_multiplier_down)
        self.insert_every = insert_every
        self.insert_n = insert_n
        self.node_calls = [0] * self.n_nodes
        self.insert_noise = insert_noise

        if dataset == 'mnist':
            self.insert_shape = (28, 28, 1)
        elif dataset == 'cifar':
            self.insert_shape = (32, 32, 3)
        else:
            raise ValueError

        np.random.seed(42)

        if insert_from is None:
            if dataset is None:
                raise ValueError
            else:
                if dataset == 'mnist':
                    mnist = MNIST()
                    self.insert_from = mnist.train_data
                elif dataset == 'cifar':
                    cifar = CIFAR()
                    self.insert_from = cifar.train_data
                else:
                    raise ValueError
        else:
            self.insert_from = insert_from

    def process_query(self) -> None:
        mapping = self.distribution_scheme.get_mapping(
            positions=[particle.position for particle in self.swarm.particles])
        for p, m in zip(self.swarm.particles, mapping):
            self.nodes[m].add_to_detector(p.position)
            self.node_calls[m] += 1
            if self.node_calls[m] == self.insert_every:
                self.node_calls[m] = 0
                if self.insert_noise is None:
                    random_indexes = np.random.choice(self.insert_from.shape[0],
                                                      size=self.insert_n)  # TRY PERLIN INSERT
                    for random_index in random_indexes:
                        random_query = self.insert_from[random_index]
                        self.nodes[m].add_to_detector(random_query)
                else:
                    if self.insert_noise == 'uniform':
                        for _ in range(self.insert_n):
                            random_query = np.random.uniform(0, 1, size=self.insert_shape)
                            self.nodes[m].add_to_detector(random_query)
                    elif self.insert_noise == 'perlin':
                        for _ in range(self.insert_n):
                            random_query = create_perlin_noise(np.array(self.insert_shape))
                            self.nodes[m].add_to_detector(random_query)
                    elif self.insert_noise == 'mixed':
                        for _ in range(self.insert_n):
                            idx = np.random.randint(0, 2)
                            if idx == 0:
                                random_query = self.insert_from[np.random.randint(0, self.insert_from.shape[0])]
                            elif idx == 1:
                                random_query = np.random.uniform(0, 1, size=self.insert_shape)
                            elif idx == 2:
                                random_query = create_perlin_noise(np.array(self.insert_shape))
                            else:
                                raise ValueError
                            self.nodes[m].add_to_detector(random_query)
                    else:
                        raise ValueError
