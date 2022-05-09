import matplotlib.pyplot as plt

from MNIST.setup_cifar import CIFAR
from MNIST.setup_mnist import MNIST


def plot_mnist_examples():
    indexes = [7270, 860, 6265, 4426, 8322]
    fig, axes = plt.subplots(1, len(indexes), figsize=(4.2, 0.9))
    mnist = MNIST()

    for i, ax in enumerate(axes):
        ax.imshow(mnist.test_data[indexes[i]], cmap='Greys_r')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()

    plt.savefig('../../Thesis/Images/mnist.png', bbox_inches='tight')


def plot_cifar_examples():
    indexes = [7270, 860, 4426, 466, 769]
    fig, axes = plt.subplots(1, len(indexes))
    cifar = CIFAR()

    for i, ax in enumerate(axes):
        ax.imshow(cifar.test_data[indexes[i]])
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()

    plt.savefig('../../Thesis/Images/cifar.png', bbox_inches='tight')


if __name__ == '__main__':
    plot_mnist_examples()
    plot_cifar_examples()
