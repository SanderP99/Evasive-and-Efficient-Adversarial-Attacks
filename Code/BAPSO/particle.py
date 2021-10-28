import numpy as np
import abc


class Particle:
    def __init__(self, id, target_image, target_label, model):
        __metaclass__ = abc.ABCMeta

        # ID to differentiate the different particle
        self.id = id
        self.target_image = target_image
        self.shape = target_image.shape
        self.target_label = target_label
        self.model = model

        # BA parameters
        self.position = None
        self.delta = 0.1
        self.eps = 1.

        # PSO parameters
        self.fitness = np.infty
        self.best_fitness = np.infty
        self.best_position = self.position

        # Initialize with no velocity
        self.velocity = np.zeros(self.shape)

    def update_position(self) -> None:
        self.position = self.position + self.velocity

    def update_best_fitness(self) -> None:
        self.calculate_fitness()
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position

    @abc.abstractmethod
    def calculate_fitness(self):
        return
