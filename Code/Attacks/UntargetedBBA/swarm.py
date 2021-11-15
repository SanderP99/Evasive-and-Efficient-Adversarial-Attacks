import numpy as np

from Test.particle import Particle


class Swarm:
    def __init__(self, n_particles, target_image, target_label, model):
        self.total_queries = 0
        self.particles = [Particle(target_image, target_label, model, self) for _ in range(n_particles)]

        self.best_position, self.best_fitness = self.get_best_particle()
        self.iteration = 0

        self.minimum_diff_constant = 1E-6
        self.number_of_iterations_before_mutation = -1
        self.number_of_iterations_between_mutations = 20
        self.previous_average = None

    def optimize(self):
        # Update particles
        self.move_swarm()

        # Update swarm
        current_best_position, current_best_fitness = self.get_best_particle()
        if current_best_fitness < self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_position = current_best_position

        self.iteration += 1
        # TODO: Prune perturbations

    def move_swarm(self):
        self.update_velocities()
        self.update_positions()
        self.update_bests()
        if self.is_stuck():
            # print("Mutation")
            self.mutate_swarm()

    def update_velocities(self):
        # FROM MGPSO
        a1, a2 = 1.0, 2.0
        for i, particle in enumerate(self.particles):
            if i % 2 == 0:
                c1, c2 = max(a1, a2), min(a1, a2)
            else:
                c1, c2 = min(a1, a2), max(a1, a2)
            particle.update_velocity(c1=c1, c2=c2)

    def update_positions(self):
        for particle in self.particles:
            particle.update_position()
            particle.generate_boundary_sample()

    def update_bests(self):
        for particle in self.particles:
            particle.update_bests()

    def mutate_swarm(self):
        for particle in self.particles:
            particle.mutate()

    def get_best_particle(self):
        best_particle = min(self.particles)
        return best_particle.position, best_particle.fitness

    def get_worst_article(self):
        worst_particle = max(self.particles)
        return worst_particle.position, worst_particle.fitness

    def is_stuck(self):
        # From MGRR-PSO
        average = np.mean([p.best_fitness for p in self.particles])
        if self.previous_average is None:
            # No change in first iteration
            self.previous_average = average
            return False

        diff = np.abs(self.previous_average - average)
        self.previous_average = average
        if self.iteration < 40 and diff == 0.0:
            return False
        if diff < self.minimum_diff_constant and diff != np.nan and self.number_of_iterations_before_mutation <= 0:
            self.minimum_diff_constant *= 0.50
            self.number_of_iterations_before_mutation = self.number_of_iterations_between_mutations
            return True
        else:
            self.number_of_iterations_before_mutation -= 1
            return False
