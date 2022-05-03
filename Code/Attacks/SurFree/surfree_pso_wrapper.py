import time

import numpy as np
from foolbox import TensorFlowModel
from keras.models import load_model

from Attacks.DistributedBBA.node import Node
from Attacks.SurFree.surfree_source import SurFree
from MNIST.setup_cifar import CIFAR
from MNIST.setup_mnist import MNIST


def get_model(dataset):
    if dataset == 'mnist':
        model = load_model('../../MNIST/models/mnist_reverse', compile=False)
        return TensorFlowModel(model, bounds=(0, 1))


if __name__ == '__main__':
    ###############################
    dataset = 'mnist'

    print("Load Model")
    fmodel = get_model(dataset)

    ###############################
    print("Load Data")
    if dataset == 'mnist':
        data = MNIST()
    elif dataset == 'cifar':
        data = CIFAR()
    else:
        raise ValueError

    images = np.array([data.test_data[42]])
    images = np.transpose(images, (0, 3, 1, 2))
    labels = np.array([np.argmax(data.test_labels[42])])
    print("{} images loaded with the following labels: {}".format(len(images), labels))

    print("Attack !")
    time_start = time.time()

    node = Node(0, 'mnist')
    f_attack = SurFree(nodes=[node])
