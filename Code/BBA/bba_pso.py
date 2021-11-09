import numpy as np

from BBA.bba import BiasedBoundaryAttack
from BBA.particle import Particle
from BBA.sampling_provider import create_perlin_noise


class ParticleBiasedBoundaryAttack:

    def __init__(self, n_particles, inits, target_img, target_label, model):
        assert n_particles == len(inits)
        self.total_queries = 0
        self.iteration = 0

        self.particles = [
            Particle(i, init=inits[i], target_img=target_img, target_label=target_label, model=model, swarm=self) for
            i in range(n_particles)]

        self.best_position, self.best_fitness = self.get_best_particle()

        self.target_img = target_img
        self.target_label = target_label

        self.model = model
        self.attack = BiasedBoundaryAttack(model, create_perlin_noise)

    def get_best_particle(self):
        best_particle = min(self.particles)
        return best_particle.position, best_particle.fitness

    def get_worst_article(self):
        worst_particle = max(self.particles)
        return worst_particle.position, worst_particle.fitness

    def update_positions(self):
        for particle in self.particles:
            particle.update_position()

    def update_bests(self):
        for particle in self.particles:
            particle.update_bests()

    def move_swarm(self):
        self.update_positions()
        self.update_bests()

    def optimize(self):
        # Update particles
        self.move_swarm()

        # Update swarm
        current_best_position, current_best_fitness = self.get_best_particle()
        if current_best_fitness < self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_position = current_best_position

        self.iteration += 1
