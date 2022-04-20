from collections import deque
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model

from Attacks.TargetedBBA.bba import BiasedBoundaryAttack
from Attacks.TargetedBBA.particle import Particle
from Attacks.TargetedBBA.sampling_provider import create_perlin_noise


class ParticleBiasedBoundaryAttack:

    def __init__(self, n_particles: int, inits: np.ndarray, target_img: np.ndarray, target_label: int, model: Model,
                 distributed_attack=None, source_step: float = 1e-2,
                 spherical_step: float = 5e-2, steps_per_iteration: int = 50, source_step_multiplier_up: float = 1.05,
                 source_step_multiplier_down: float = 0.6, use_node_manager: bool = False, dataset=None,
                 distance_based: Optional[str] = None, history_len=10):
        assert n_particles == len(inits)
        self.n_particles: int = n_particles
        self.total_queries: int = 0
        self.iteration: int = 0

        self.target_img: np.ndarray = target_img
        self.target_label: int = target_label

        self.model: Model = model
        distributed: bool = True if distributed_attack is not None else False
        self.attack: BiasedBoundaryAttack = BiasedBoundaryAttack(model, create_perlin_noise)

        if distributed_attack is not None:
            self.distributed_attack = distributed_attack
            self.nodes: list = self.distributed_attack.nodes
            self.distribution_scheme = self.distributed_attack.distribution_scheme
            # self.mapping: deque = self.distributed_attack.distribution_scheme.get_mapping()

        self.particles: list = [
            Particle(i, init=inits[i], target_img=target_img, target_label=target_label, model=model, swarm=self,
                     source_step_multiplier_up=source_step_multiplier_up,
                     source_step_multiplier_down=source_step_multiplier_down, source_step=source_step,
                     spherical_step=spherical_step, steps_per_iteration=steps_per_iteration,
                     use_node_manager=use_node_manager, dataset=dataset, distance_based=distance_based,
                     history_len=history_len) for
            i in range(n_particles)]

        self.best_position, self.best_fitness, _ = self.get_best_particle()

    def get_best_particle(self) -> (np.ndarray, float):
        best_particle = min(self.particles)
        return best_particle.position, best_particle.fitness, best_particle.current_label

    def get_worst_article(self) -> (np.ndarray, float):
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

    def optimize(self) -> None:
        # Update particles
        self.move_swarm()

        # Update swarm
        current_best_position, current_best_fitness, _ = self.get_best_particle()

        # if self.iteration % 100 == 0:
        #     fig, ax = plt.subplots(2)
        #     ax[0].imshow(self.best_position)
        #     ax[0].set_title(str(self.best_fitness))
        #     ax[1].imshow(self.target_img)
        #     ax[1].set_title(self.target_label)
        #     plt.show()

        if current_best_fitness < self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_position = current_best_position

        self.iteration += 1
