import numpy as np
from keras.models import load_model

model = load_model('../MNIST/models/mnist', compile=False)


def initialize_position(shape):
    return np.random.random(shape).flatten()


def fitness_function(position, target_image, shape, target_label, targeted):
    distance = np.linalg.norm(target_image - position)
    prediction = np.argmax(model.predict(position.reshape((1,) + shape)))
    print(prediction)
    if targeted:
        # Label needs to be the same as target
        if target_label == prediction:
            adversarial_criterion = 0
        else:
            adversarial_criterion = np.infty
    else:
        # Label needs to be different from target
        if target_label != prediction:
            adversarial_criterion = 0
        else:
            adversarial_criterion = np.infty

    return distance + adversarial_criterion


class Particle:
    def __init__(self, target_image, target_label, targeted, init=None):
        if init is not None:
            self.position = init
        else:
            self.position = initialize_position(target_image.shape)
        self.shape = target_image.shape
        self.velocity = [0] * np.prod(target_image.shape)
        self.personal_best_position = self.position
        self.target_label = target_label
        self.targeted = targeted
        self.target_image = target_image.flatten()
        self.personal_best_fitness = self.calculate_fitness()
        print(self.target_label, self.personal_best_fitness)

    def __str__(self):
        return f"Particle at {self.position} \n with velocity {self.velocity}. Current best: {self.personal_best_fitness}"

    def calculate_fitness(self):
        return fitness_function(self.position, self.target_image, self.shape, self.target_label, self.targeted)

    def update_position(self, swarm_best_position):
        pass
