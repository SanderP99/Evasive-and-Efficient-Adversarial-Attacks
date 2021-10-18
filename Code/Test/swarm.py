import numpy as np

from Test.particle import Particle


class Swarm:
    def __init__(self, n_particles, target_image, target_label, model):
        self.particles = [Particle(target_image, target_label, model, self) for i in range(n_particles)]

        self.best_position, self.best_fitness = self.get_best_particle()
        self.iteration = 0

    def optimize(self):
        # Update particles
        self.move_swarm()

        # Update swarm
        current_best_position, current_best_fitness = self.get_best_particle()
        if current_best_fitness < self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_position = current_best_position

        self.iteration += 1

    def move_swarm(self):
        self.update_velocities()
        self.update_positions()
        self.update_bests()

    def update_velocities(self):
        for particle in self.particles:
            particle.update_velocity()

    def update_positions(self):
        for particle in self.particles:
            particle.update_position()

    def update_bests(self):
        for particle in self.particles:
            particle.update_bests()

    def get_best_particle(self):
        best_particle = min(self.particles)
        return best_particle.position, best_particle.fitness
