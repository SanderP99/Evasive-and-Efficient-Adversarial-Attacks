from dataclasses import dataclass

from MNIST.setup_cifar import CIFAR
from MNIST.setup_mnist import MNIST


@dataclass
class InsertNoise:
    insert_n: int
    insert_every: int
    insert_noise: str
    insert_from: str

    def __init__(self, insert_n, insert_every, insert_noise, insert_from):
        self.insert_n = insert_n
        self.insert_every = insert_every
        self.insert_noise = insert_noise

        if insert_from == 'mnist':
            mnist = MNIST()
            self.insert_from = mnist.test_data
            self.insert_shape = (28, 28, 1)
        elif insert_from == 'cifar':
            cifar = CIFAR()
            self.insert_from = cifar.test_data
            self.insert_shape = (32, 32, 3)
        else:
            raise ValueError
