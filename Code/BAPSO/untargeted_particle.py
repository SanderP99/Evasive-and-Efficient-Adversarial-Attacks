from BAPSO.particle import Particle
import numpy as np


class UntargetedParticle(Particle):
    def __init__(self, id, target_image, target_label, model):
        super().__init__(id, target_image, target_label, model)

        # Initialize randomly in untargeted attack
        self.position = np.random.random(self.shape).flatten()
        self.best_position = self.position

    def calculate_fitness(self) -> None:
        prediction = np.argmax(self.model.predict(self.position.reshape((1,) + self.shape)))
        print("Prediction: ", prediction)
        if prediction == self.target_label:
            # No longer adversarial
            self.fitness = np.infty
        else:
            # Still adversarial, fitness is L2 distance
            self.fitness = np.linalg.norm(self.target_image - self.position)

    def update_velocity(self, swarm_best_position: np.array, c1=0., c2=0.) -> None:
        distance_from_personal_best = self.best_position - self.position
        distance_from_swarm_best = swarm_best_position - self.position

        target_image_direction = self.target_image.flatten() - self.position
        norm = np.linalg.norm(target_image_direction)
        target_image_direction /= norm

        orthogonal_direction = self.orthogonal_perturbation()

        c3 = self.delta
        c4 = self.eps
        self.velocity = c1 * np.random.random(self.shape).flatten() * distance_from_personal_best + \
                        c2 * np.random.random(self.shape).flatten() * distance_from_swarm_best + \
                        orthogonal_direction + c4 * target_image_direction

    def orthogonal_perturbation(self) -> np.array:
        # From https://github.com/greentfrapp/boundary-attack/blob/master/boundary-attack-resnet.py
        # Generate perturbation
        perturb = np.random.random(self.shape).flatten()
        perturb /= np.linalg.norm(perturb)
        perturb *= self.delta * np.linalg.norm(self.target_image.flatten() - self.position)
        # Project perturbation onto sphere around target
        diff = (self.target_image.flatten() - self.position).astype(np.float32)
        diff /= np.linalg.norm(diff)
        perturb -= np.dot(perturb, diff) * diff
        # Check overflow and underflow
        overflow = (self.position + perturb) - np.ones_like(self.position)
        perturb -= overflow * (overflow > 0)
        underflow = np.zeros_like(self.position) - (self.position + perturb)
        perturb += underflow * (underflow > 0)
        return perturb
