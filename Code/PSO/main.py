import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np

from MNIST.setup_mnist import MNIST
from PSO.swarm import Swarm

if __name__ == '__main__':
    mnist = MNIST()
    model = load_model('../MNIST/models/mnist', compile=False)
    target_image = mnist.test_data[0]
    label = np.argmax(model.predict(target_image.reshape(1, 28, 28, 1)))
    plt.imshow(target_image, cmap=plt.get_cmap('gray'))
    plt.show()
    swarm = Swarm(target_image=target_image, target_label=label, model=model)
    # plt.imshow(swarm.particles[0].position.reshape((28, 28, 1)), cmap=plt.get_cmap('gray'))
    # plt.show()
    for i in range(1000):
        swarm.optimize()
        if i % 40 == 0:
            print(swarm.current_best_fitness)
            fig, ax = plt.subplots(1, 3)
            fig.suptitle(f'Iteration {i}')
            ax[0].set_title('Best particle')
            ax[1].set_title('Worst particle')
            ax[2].set_title('Difference')
            ax[0].imshow(swarm.particles[0].position.reshape((28, 28, 1)), cmap=plt.get_cmap('gray'))
            ax[1].imshow(swarm.particles[-1].position.reshape((28, 28, 1)), cmap=plt.get_cmap('gray'))
            ax[2].imshow((swarm.particles[0].position - swarm.particles[-1].position).reshape((28, 28, 1)), cmap=plt.get_cmap('gray'))
            prediction = np.argmax(model.predict(swarm.particles[0].position.reshape((1, 28, 28, 1))))
            print(prediction)
            plt.show()
