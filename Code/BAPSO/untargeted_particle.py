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
        if prediction == self.target_label:
            # No longer adversarial
            self.fitness = np.infty
        else:
            # Still adversarial, fitness is L2 distance
            self.fitness = np.linalg.norm(self.target_image - self.position)

    def update_velocity(self, swarm_best_position: np.array, c1=0., c2=0.) -> None:

        # Orthogonal step
        while True:
            trial_samples = []
            for _ in np.arange(10):
                trial_sample = self.position + self.orthogonal_perturbation()
                trial_samples.append(trial_sample)
            predictions = self.model.predict(np.array(trial_samples).reshape((10,) + self.shape))
            predictions = np.argmax(predictions, axis=1)
            d_score = np.mean(predictions != self.target_label)
            if d_score > 0.0:
                if d_score < 0.3:
                    self.delta *= 0.9
                elif d_score > 0.7:
                    self.delta /= 0.9
                adversarial_sample = np.array(trial_samples)[np.where(predictions != self.target_label)[0][0]]
                break
            else:
                self.delta *= 0.9
        # Forward step
        e_step = 0
        while True:
            e_step += 1
            # print("\t#{}".format(e_step))
            trial_sample = adversarial_sample + self.forward_perturbation()
            prediction = self.model.predict(trial_sample.reshape((1,) + self.shape))
            if np.argmax(prediction) != self.target_label:
                adversarial_sample = trial_sample
                self.eps /= 0.5
                break
            elif e_step > 500:
                break
            else:
                self.eps *= 0.5

        self.velocity = adversarial_sample - self.position

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

    def forward_perturbation(self):
        perturb = (self.target_image.flatten() - self.position)
        perturb *= self.eps
        return perturb
