from functools import total_ordering

import numpy as np

from BBA.utils import line_search_to_boundary


@total_ordering
class Particle:

    def __init__(self, i, init=None, target_img=None, target_label=0, model=None, swarm=None, is_targeted=True):
        self.id = i
        self.position = init
        self.position = line_search_to_boundary(model, target_img, self.position, target_label, True)
        self.is_adversarial = True
        self.is_targeted = is_targeted

        self.target_image = target_img
        self.target_label = target_label

        self.model = model
        self.swarm = swarm

        self.steps_per_iteration = 20
        self.fitness = np.infty
        self.calculate_fitness()
        self.best_position = self.position
        self.best_fitness = self.fitness

        self.source_step = 0.02
        self.spherical_step = 5e-2

    def __eq__(self, other):
        return self.fitness == other.fitness and self.best_fitness == other.best_fitness

    def __lt__(self, other):
        return self.fitness < other.fitness or (
                self.fitness == other.fitness and self.best_fitness < other.best_fitness)

    def calculate_fitness(self):
        prediction = np.argmax(self.model.predict(np.expand_dims(self.position, axis=0)))
        self.swarm.total_queries += 1

        if (prediction == self.target_label) != self.is_targeted:
            # No longer adversarial
            self.fitness = np.infty
            self.is_adversarial = False
        else:
            self.fitness = np.linalg.norm(self.position - self.target_image)
            self.is_adversarial = True

    def update_position(self):
        mask = np.abs(self.position - self.target_image)
        mask /= np.max(mask)  # scale to [0,1]
        mask = mask ** 0.5  # weaken the effect a bit.
        self.position = self.swarm.attack.run_attack(self.target_image, self.target_label, True, self.position,
                                                     (lambda: self.steps_per_iteration - self.swarm.attack.calls),
                                                     source_step=self.source_step,
                                                     spherical_step=self.spherical_step, mask=mask)
        self.swarm.total_queries += self.swarm.attack.calls

    def update_bests(self):
        self.calculate_fitness()
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position
