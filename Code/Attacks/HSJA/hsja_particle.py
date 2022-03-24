from functools import total_ordering

import numpy as np
from keras.models import Model

from Attacks.HSJA.hsja_position import HSJAPosition


@total_ordering
class HSJAParticle:
    def __init__(self, i: int, init: HSJAPosition = None, target_img: np.ndarray = None, target_label: int = 0,
                 model: Model = None, swarm=None):
        self.id: int = i
        self.position: HSJAPosition = init
        self.velocity: HSJAPosition = HSJAPosition()

        self.target_image: np.ndarray = target_img
        self.target_label: int = target_label

        self.model: Model = model
        self.swarm = swarm

        self.fitness: float = np.infty
        self.best_position: HSJAPosition = self.position
        self.best_fitness: float = self.fitness

        self.c1 = 2.
        self.c2 = 2.
        self.w: HSJAPosition = HSJAPosition()

    def __eq__(self, other: 'HSJAParticle') -> bool:
        return self.fitness == other.fitness and self.best_fitness == other.best_fitness

    def __lt__(self, other: 'HSJAParticle') -> bool:
        return self.fitness < other.fitness or (
                self.fitness == other.fitness and self.best_fitness < other.best_fitness)

    def update_position(self) -> None:
        self.update_velocity()
        self.position += self.velocity

    def update_velocity(self) -> None:
        particle_best_delta = self.c2 * ((self.best_position - self.position) * self.random_position())
        swarm_best_delta = self.c1 * ((self.swarm.best_position - self.position) * self.random_position())
        deltas = particle_best_delta + swarm_best_delta
        self.velocity = self.w * deltas

    @staticmethod
    def random_position():
        init_n_evals = np.random.uniform(0., 1.)
        max_n_evals = np.random.uniform(0., 1.)
        step_size_decrease = np.random.uniform(0., 1.)
        gamma = np.random.uniform(0., 1.)
        return HSJAPosition(init_n_evals=init_n_evals, max_n_evals=max_n_evals, step_size_decrease=step_size_decrease,
                            gamma=gamma)
