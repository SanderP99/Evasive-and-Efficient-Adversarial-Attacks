from PSO.particle import Particle


class Swarm:
    def __init__(self, target_image, n_particles=20, target_label=0, targeted=False, model=None):
        self.particles = [
            Particle(i, target_image=target_image, target_label=target_label, targeted=targeted, model=model) for i in
            range(n_particles)]

        # Sort based on fitness values
        self.particles.sort(reverse=True)
        self.swarm_best_position = self.particles[0].personal_best_position
        self.swarm_best_fitness = self.particles[0].personal_best_fitness
        self.current_best_fitness = self.particles[0].personal_best_fitness
        self.current_best_position = self.particles[0].personal_best_position

    def __str__(self):
        return f"Swarm containing {len(self.particles)} particles. Current best particle: ID={self.particles[0].id} with {self.swarm_best_fitness}"

    def optimize(self):
        # Update all particles
        for particle in self.particles:
            particle.update_velocity(self.swarm_best_position, c1=.2, c2=.2)
            particle.update_position()
            particle.update_personal_best()

        # Update swarm
        self.particles.sort(reverse=True)
        if self.particles[0].fitness < self.swarm_best_fitness:
            self.swarm_best_position = self.particles[0].personal_best_position
            self.swarm_best_fitness = self.particles[0].personal_best_fitness

        self.current_best_fitness = self.particles[0].fitness
        self.current_best_position = self.particles[0].position

