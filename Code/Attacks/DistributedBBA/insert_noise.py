from dataclasses import dataclass

from MNIST.setup_cifar import CIFAR
from MNIST.setup_mnist import MNIST


@dataclass
class InsertNoise:
    insert_n: int
    insert_every: int
    insert_noise: str
    insert_from: str
    decay: bool

    def __init__(self, insert_n, insert_every, insert_noise, insert_from, decay=False, n_nodes=5):
        self.insert_n = insert_n
        self.insert_every = insert_every
        self.insert_noise = insert_noise
        self.decay = decay
        self.decay_rate = self.insert_every
        self.n_nodes = n_nodes

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

    def set_decay_rate(self, node_calls):
        self.decay_rate = int(((25000 - self.n_nodes * node_calls) / 25000) * self.insert_every)

    def reset(self):
        self.decay_rate = self.insert_every
