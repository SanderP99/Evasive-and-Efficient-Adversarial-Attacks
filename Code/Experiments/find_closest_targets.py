import pandas as pd
import numpy as np
import ast

from MNIST.setup_cifar import CIFAR
from MNIST.setup_mnist import MNIST

orig = None


def distance(target):
    return np.linalg.norm(orig - cifar.test_data[target])


if __name__ == '__main__':
    df = pd.read_csv('experiments_cifar.csv', index_col='index')
    sorted_i = []
    cifar = CIFAR()
    for i, data in df.iterrows():
        orig = cifar.test_data[i]
        target_indices = ast.literal_eval(data['targets'])
        sorted_indices = sorted(target_indices, key=distance)
        sorted_i.append(sorted_indices)
    df['sorted_targets'] = sorted_i
    df.to_csv('experiments_cifar_sorted.csv')

