import ast
from functools import total_ordering

import numpy as np
import pandas as pd
from keras.models import Model

from Attacks.HSJA.hsja_position import HSJAPosition
from Attacks.HSJA.hsja_vanilla import hsja
from MNIST.setup_mnist import MNIST


@total_ordering
class HSJAParticle:
    def __init__(self, i: int, init: HSJAPosition = None,
                 model: Model = None, swarm=None):
        self.id: int = i
        self.position: HSJAPosition = init
        self.velocity: HSJAPosition = HSJAPosition()
        self.experiment_idx: int = 0
        self.experiments = pd.read_csv('../../Experiments/experiments_sorted.csv', index_col='index')

        self.model: Model = model
        self.swarm = swarm

        self.fitness: float = np.infty
        self.best_position: HSJAPosition = self.position
        self.best_fitness: float = self.fitness

        self.c1 = 2.
        self.c2 = 2.
        self.w: HSJAPosition = HSJAPosition()
        self.mnist = MNIST()
        self.queries = 0


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

    def update_bests(self) -> None:
        self.calculate_fitness()
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position

    def calculate_fitness(self) -> None:
        experiment = self.experiments.iloc[self.experiment_idx]
        x_orig = self.mnist.test_data[experiment.name]
        targets = ast.literal_eval(experiment.targets)
        random_inits = self.mnist.test_data[
            np.array(targets)[np.random.choice(len(targets), size=1, replace=False)]]
        # self.experiment_idx += 1
        _, qdw = hsja(self.model, np.expand_dims(x_orig, axis=0), target_label=experiment.y_target,
                      target_image=random_inits[0], distributed=True,
                      flush_buffer_after_detection=True,
                      init_num_evals=self.position.init_n_evals,
                      max_num_evals=self.position.max_n_evals,
                      gamma=self.position.gamma,
                      step_size_decrease=self.position.step_size_decrease)
        queries, detections = qdw.n_queries, qdw.get_n_detections()
        self.queries = queries
        self.fitness = detections / queries
