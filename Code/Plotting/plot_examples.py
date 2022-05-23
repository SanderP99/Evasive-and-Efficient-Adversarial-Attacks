import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from MNIST.setup_cifar import CIFAR
from MNIST.setup_mnist import MNIST

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


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
    indexes = [7270, 860, 5734, 466, 4426]
    fig, axes = plt.subplots(1, len(indexes))
    cifar = CIFAR()

    for i, ax in enumerate(axes):
        ax.imshow(cifar.test_data[indexes[i]])
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()

    plt.savefig('../../Thesis/Images/cifar.png', bbox_inches='tight')


def plot_comparison_mnist():
    indexes = [7270, 860, 6265, 4426, 8322]
    fig, axes = plt.subplots(3, len(indexes), figsize=(4.2, 3 * 0.9))
    mnist = MNIST()
    experiments = pd.read_csv('../Experiments/experiments_sorted.csv', index_col='index')

    for i, ax in enumerate(axes):
        for j, a in enumerate(ax):
            if i == 0:
                a.imshow(mnist.test_data[indexes[j]], cmap='Greys_r')
                a.set_xticks([])
                a.set_yticks([])
                a.set_title(rf'{np.argmax(mnist.test_labels[indexes[j]])}', fontsize=7)
            elif i == 1:
                orig = mnist.test_data[indexes[j]]
                with open(f'../Experiments/DistributedBBA/result_images/pso_bba_mnist_{j}.npy', 'rb') as f:
                    x = np.load(f)
                dist = np.around(np.linalg.norm(x - orig), decimals=3)
                dist = "{:0.3f}".format(dist)
                a.imshow(x, cmap='Greys_r')
                a.set_xticks([])
                a.set_yticks([])
                a.set_title(rf'{experiments.iloc[j].y_target} ({dist})', fontsize=7)
            elif i == 2:
                orig = mnist.test_data[indexes[j]]
                with open(f'../Experiments/DistributedBBA/result_images/hsja_mnist_{j}.npy', 'rb') as f:
                    x = np.load(f)[0]
                dist = np.round(np.linalg.norm(x - orig), 3)
                a.imshow(x, cmap='Greys_r')
                a.set_xticks([])
                a.set_yticks([])
                a.set_title(rf'{experiments.iloc[j].y_target} ({dist})'.capitalize(), fontsize=7)
    fig.tight_layout()
    plt.savefig('../../Thesis/Images/mnist_comparison.pdf', bbox_inches='tight', format='pdf')
    plt.show()


def plot_comparison_cifar():
    indexes = [7270, 860, 5734, 466, 4426]
    names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    fig, axes = plt.subplots(3, len(indexes), figsize=(4.2, 3 * 0.9))
    mnist = CIFAR()
    experiments = pd.read_csv('../Experiments/experiments_cifar_sorted2.csv', index_col='index')

    for i, ax in enumerate(axes):
        for j, a in enumerate(ax):
            if i == 0:
                a.imshow(mnist.test_data[indexes[j]])
                a.set_xticks([])
                a.set_yticks([])
                a.set_title(f'{names[np.argmax(mnist.test_labels[indexes[j]])]}'.capitalize(), fontsize=7)
            elif i == 1:
                orig = mnist.test_data[indexes[j]]
                with open(f'../Experiments/DistributedBBA/result_images/pso_bba_cifar_{j}.npy', 'rb') as f:
                    x = np.load(f)
                dist = np.around(np.linalg.norm(x - orig), decimals=3)
                dist = "{:0.3f}".format(dist)
                a.imshow(x)
                a.set_xticks([])
                a.set_yticks([])
                a.set_title(f'{names[experiments.iloc[j].y_target]} ({dist})'.capitalize(), fontsize=7)
            elif i == 2:
                orig = mnist.test_data[indexes[j]]
                with open(f'../Experiments/DistributedBBA/result_images/hsja_cifar_{j}.npy', 'rb') as f:
                    x = np.load(f)[0]
                dist = np.round(np.linalg.norm(x - orig), 3)
                a.imshow(x)
                a.set_xticks([])
                a.set_yticks([])
                a.set_title(f'{names[experiments.iloc[j].y_target]} ({dist})'.capitalize(), fontsize=7)
    fig.tight_layout()
    plt.savefig('../../Thesis/Images/cifar_comparison.pdf', bbox_inches='tight', format='pdf')
    plt.show()


if __name__ == '__main__':
    # plot_mnist_examples()
    # plot_cifar_examples()
    # plot_comparison_mnist()
    plot_comparison_cifar()
