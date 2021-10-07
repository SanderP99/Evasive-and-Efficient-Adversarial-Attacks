from PSO.particle import Particle


class Swarm:
    def __init__(self, target_image, n_particles=20, target_label=0, targeted=False):
        self.particles = [Particle(target_image=target_image, target_label=target_label, targeted=targeted) for _ in
                          range(n_particles)]
        self.swarm_best_position = None
        self.swarm_best_fitness = None

    def __str__(self):
        return f"Swarm containing {len(self.particles)} particles. Current best particle: {self.swarm_best_fitness}"
