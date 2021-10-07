import numpy as np


def initialize_position(shape):
    return np.random.random(shape).flatten()


class Particle:
    def __init__(self, id, target_image, target_label, targeted, model, init=None):
        self.id = id
        if init is not None:
            self.position = init
        else:
            self.position = initialize_position(target_image.shape)
        self.shape = target_image.shape
        self.velocity = np.random.normal(0, 0.1, target_image.shape).flatten()
        self.personal_best_position = self.position
        self.target_label = target_label
        self.targeted = targeted
        self.target_image = target_image.flatten()
        self.model = model
        self.personal_best_fitness = self.calculate_fitness()

    def __str__(self):
        return f"Particle {self.id} at {self.position} \n with velocity {self.velocity}. Current best: {self.personal_best_fitness}"

    def __gt__(self, other):
        """
        Greater than functionality based on the personal best fitness values. Useful for sorting
        :param other: The particle to compare to
        """
        return self.personal_best_fitness < other.personal_best_fitness

    # Indirection in case different fitness functions need to be tested
    def calculate_fitness(self):
        return self.fitness_function()

    def fitness_function(self):
        distance = np.linalg.norm(self.target_image - self.position)
        prediction = np.argmax(self.model.predict(self.position.reshape((1,) + self.shape)))
        if self.targeted:
            # Label needs to be the same as target
            if self.target_label == prediction:
                adversarial_criterion = 0
            else:
                adversarial_criterion = np.infty
        else:
            # Label needs to be different from target
            if self.target_label != prediction:
                adversarial_criterion = 0
            else:
                adversarial_criterion = np.infty
        return distance + adversarial_criterion

    def update_position(self):
        self.position = np.clip(self.position + self.velocity, 0, 1)

    def update_velocity(self, swarm_best_position, c1=2, c2=2):
        distance_from_personal_best = self.personal_best_position - self.position
        distance_from_swarm_best = swarm_best_position - self.position
        self.velocity = self.velocity + c1 * np.random.random(self.shape).flatten() * distance_from_personal_best \
                        + c2 * np.random.random(self.shape).flatten() * distance_from_swarm_best

    def update_personal_best(self):
        fitness = self.calculate_fitness()
        if fitness < self.personal_best_fitness:
            self.personal_best_fitness = fitness
            self.personal_best_position = self.position
