from Attacks.TargetedBBA.bba import BiasedBoundaryAttack
from Attacks.TargetedBBA.particle import Particle
from Attacks.TargetedBBA.sampling_provider import create_perlin_noise


class ParticleBiasedBoundaryAttack:

    def __init__(self, n_particles, inits, target_img, target_label, model, distributed_attack=None):
        assert n_particles == len(inits)
        self.total_queries = 0
        self.iteration = 0

        self.target_img = target_img
        self.target_label = target_label

        self.model = model
        distributed = True if distributed_attack is not None else False
        self.attack = BiasedBoundaryAttack(model, create_perlin_noise)

        if distributed_attack is not None:
            self.distributed_attack = distributed_attack
            self.nodes = self.distributed_attack.nodes
            self.distribution_scheme = self.distributed_attack.distribution_scheme
            self.mapping = self.distributed_attack.mapping

        self.particles = [
            Particle(i, init=inits[i], target_img=target_img, target_label=target_label, model=model, swarm=self) for
            i in range(n_particles)]

        self.best_position, self.best_fitness = self.get_best_particle()

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
