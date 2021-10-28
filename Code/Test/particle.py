import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

from functools import total_ordering

from Test.perlin import create_perlin_noise


@total_ordering
class Particle:
    def __init__(self, target_image, target_label, model, swarm):
        # self.position = np.random.random(target_image.shape)
        self.position = create_perlin_noise(target_image.shape)

        self.target_image = target_image
        self.target_label = target_label

        self.model = model
        self.swarm = swarm

        self.fitness = np.infty
        self.calculate_fitness()

        self.velocity = np.zeros_like(self.position)
        self.maximum_diff = 0.4
        self.mutation_rate = 0.5

        self.best_position = self.position
        self.best_fitness = self.fitness

        self.iteration = 0

    def __eq__(self, other):
        return self.fitness == other.fitness and self.best_fitness == other.best_fitness

    def __lt__(self, other):
        return self.fitness < other.fitness or (
                self.fitness == other.fitness and self.best_fitness < other.best_fitness)

    def calculate_fitness(self):
        prediction = np.argmax(self.model.predict(np.expand_dims(self.position, axis=0)))
        self.swarm.total_queries += 1

        if prediction == self.target_label:
            # No longer adversarial
            self.fitness = np.infty
        else:
            self.fitness = np.linalg.norm(self.position - self.target_image)

    def update_position(self):
        new_position = np.add(self.position, self.velocity)
        self.position = np.clip(new_position, 0, 1)
        self.iteration += 1

    def update_velocity(self, c1=2., c2=2., c3=0.4, c4=0.4):
        w = self.calculate_w(1., 0., 1000)
        particle_best_delta = c2 * (self.best_position - self.position) * np.random.uniform(0., 1.,
                                                                                            self.target_image.shape)
        swarm_best_delta = c1 * (self.swarm.best_position - self.position) * np.random.uniform(0., 1.,
                                                                                               self.target_image.shape)
        target_delta = c3 * (self.target_image - self.position) * np.random.uniform(0., 1., self.target_image.shape)
        # TODO: binary search to find boundary?
        perturb = np.random.random(self.target_image.shape)
        perturb /= np.linalg.norm(perturb)

        # TODO: Use perlin noise to go in low freq direction
        # Project onto sphere around target
        diff = self.target_image - self.position
        diff /= np.linalg.norm(diff)

        perturb -= (np.vdot(perturb, diff) / np.linalg.norm(diff) ** 2) * diff
        perturb = np.clip(perturb, 0., 1.)

        orthogonal_delta = c4 * np.random.uniform(0., 1., self.target_image.shape) * perturb
        deltas = particle_best_delta + swarm_best_delta + target_delta + orthogonal_delta
        self.velocity = np.add(w * np.clip(self.velocity, -1 * self.maximum_diff, self.maximum_diff), deltas)

        # TODO: Fix masking
        # mask = np.abs(self.target_image - self.position)
        #
        # self.velocity *= mask

    def calculate_w(self, w_start, w_end, max_queries):
        if np.all(np.equal(self.best_position, self.swarm.best_position)):
            return w_end
        else:
            return w_end + ((w_start - w_end) * (1 - (self.iteration / max_queries)))

    def update_bests(self):
        self.calculate_fitness()
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position

    def mutate(self):
        if np.random.random() < self.mutation_rate:
            # Only mutate a small amount of particles
            random_perturbations = np.random.random(self.target_image.shape)
            indices = np.random.random(self.target_image.shape)
            percentage = 0.5
            indices[indices < percentage] = 1
            indices[indices >= percentage] = 0
            random_perturbations *= indices
            self.position -= np.clip(indices * 2, 0, 1)
            self.position = self.position + random_perturbations
            self.velocity = np.zeros_like(self.velocity)
            self.update_bests()
