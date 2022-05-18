from dataclasses import dataclass
from typing import Optional

import numpy as np

from MNIST.setup_cifar import CIFAR
from MNIST.setup_mnist import MNIST


@dataclass
class Experiment:
    id: int
    x_orig: np.ndarray
    inits: np.ndarray
    y_target: int
    y_orig: int
    dataset: str
    random_inits: Optional[np.ndarray] = None

    def set_random_inits(self, n):
        if self.dataset == 'mnist':
            data = MNIST()
        elif self.dataset == 'cifar':
            data = CIFAR()
        else:
            raise ValueError
        self.random_inits = data.test_data[
            np.array(self.inits)[np.random.choice(len(self.inits), size=n, replace=False)]]
