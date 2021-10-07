import matplotlib.pyplot as plt

from MNIST.setup_mnist import MNIST
from PSO.swarm import Swarm

if __name__ == '__main__':
    mnist = MNIST()
    swarm = Swarm(mnist.test_data[0])
    print(swarm)
    plt.imshow(swarm.particles[0].position.reshape((28, 28, 1)), cmap=plt.get_cmap('gray'))
    plt.show()
    print(swarm.particles[0].personal_best_fitness)
