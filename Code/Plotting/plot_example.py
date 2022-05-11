import matplotlib.pyplot as plt
import numpy as np

from MNIST.setup_mnist import MNIST


def plot_example(digit, index):
    mnist = MNIST()
    np.random.seed(index)
    examples = mnist.train_data[np.argmax(mnist.train_labels, axis=1) == digit]
    example = examples[np.random.randint(examples.shape[0])]
    fig, ax = plt.subplots()
    ax.imshow(example, cmap='Greys_r')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    plt.savefig(f'../../Thesis/Images/mnist_{digit}_{index}.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    plot_example(2, 0)
