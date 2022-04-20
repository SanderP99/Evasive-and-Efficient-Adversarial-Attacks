from typing import List

import numpy as np
from keras.models import Model

from Attacks.HSJA.hsja_particle import HSJAParticle
from Attacks.HSJA.hsja_position import HSJAPosition


class HSJASwarm:

    def __init__(self, n_particles: int, inits: List[HSJAPosition],
                 model: Model):
        assert n_particles == len(inits)
        self.n_particles: int = n_particles
        self.total_queries: int = 0
        self.iteration: int = 0

        self.model: Model = model

        self.particles: List[HSJAParticle] = [
            HSJAParticle(i, init=inits[i], model=model, swarm=self)
            for i in
            range(self.n_particles)]

        self.best_position, self.best_fitness = self.get_best_particle()

    def get_best_particle(self) -> (HSJAPosition, float):
        best_particle = min(self.particles)
        return best_particle.position, best_particle.fitness

    def get_worst_article(self) -> (HSJAPosition, float):
        worst_particle = max(self.particles)
        return worst_particle.position, worst_particle.fitness

    def update_positions(self) -> None:
        for particle in self.particles:
            particle.update_position()

    def update_bests(self) -> None:
        for particle in self.particles:
            particle.update_bests()

    def move_swarm(self) -> None:
        self.update_positions()
        self.update_bests()
        self.total_queries += np.sum([particle.queries for particle in self.particles])

    def optimize(self) -> None:
        self.move_swarm()
        current_best_position, current_best_fitness = self.get_best_particle()

        if current_best_fitness < self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_position = current_best_position
        self.iteration += 1
