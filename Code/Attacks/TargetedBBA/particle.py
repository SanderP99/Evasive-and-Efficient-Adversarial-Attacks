from functools import total_ordering

import numpy as np

from Attacks.TargetedBBA.utils import line_search_to_boundary


@total_ordering
class Particle:

    def __init__(self, i, init=None, target_img=None, target_label=0, model=None, swarm=None, is_targeted=True):
        self.id = i
        self.position = init
        self.velocity = np.zeros_like(self.position)
        self.position, calls = line_search_to_boundary(model, target_img, self.position, target_label, True, True)
        self.is_adversarial = True
        self.is_targeted = is_targeted

        self.target_image = target_img
        self.target_label = target_label

        self.model = model
        self.swarm = swarm
        self.swarm.total_queries += calls

        self.steps_per_iteration = 50
        self.fitness = np.infty
        self.calculate_fitness()
        self.best_position = self.position
        self.best_fitness = self.fitness

        self.source_step = 0.25
        self.spherical_step = 5e-2
        self.maximum_diff = 0.4
        self.c1, self.c2 = self.select_cs()

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
        if self.is_adversarial:
            mask = np.abs(self.position - self.target_image)
            mask /= np.max(mask)  # scale to [0,1]
            mask = mask ** 0.5  # weaken the effect a bit.
            self.position = self.swarm.attack.run_attack(self.target_image, self.target_label, True, self.position,
                                                         (lambda: self.steps_per_iteration - self.swarm.attack.calls),
                                                         source_step=self.source_step,
                                                         spherical_step=self.spherical_step, mask=mask, pso=True)
            self.swarm.total_queries += self.swarm.attack.calls
            self.source_step *= 1.05
        else:
            # print(f"Particle {self.id} is not longer adversarial!")
            self.update_velocity()
            self.position += self.velocity
            self.source_step *= 0.6

    def update_bests(self):
        self.calculate_fitness()
        # print(f"Particle {self.id}: {self.fitness}")
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position

    def update_velocity(self, c1=2., c2=2.):
        w = self.calculate_w(1., 0., 1000)
        particle_best_delta = c2 * (self.best_position - self.position) * np.random.uniform(0., 1.,
                                                                                            self.target_image.shape)
        swarm_best_delta = c1 * (self.swarm.best_position - self.position) * np.random.uniform(0., 1.,
                                                                                               self.target_image.shape)
        deltas = particle_best_delta + swarm_best_delta
        self.velocity = w * np.clip(self.velocity + deltas, -1 * self.maximum_diff, self.maximum_diff)

    def calculate_w(self, w_start, w_end, max_queries):
        if np.all(np.equal(self.best_position, self.swarm.best_position)):
            return w_end
        else:
            return w_end + ((w_start - w_end) * (1 - (self.swarm.iteration / max_queries)))

    def select_cs(self):
        a1, a2 = 1., 2.
        if self.id % 2 == 0:
            c1, c2 = max(a1, a2), min(a1, a2)
        else:
            c1, c2 = min(a1, a2), max(a1, a2)
        return c1, c2
