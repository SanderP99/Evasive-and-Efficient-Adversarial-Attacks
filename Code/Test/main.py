import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np

from MNIST.setup_mnist import MNIST
from Test.swarm import Swarm

if __name__ == '__main__':
    mnist = MNIST()
    model = load_model('../MNIST/models/mnist', compile=False)
    target_image = mnist.test_data[42]
    label = np.argmax(model.predict(target_image.reshape(1, 28, 28, 1)))
    current_best = np.infty
    swarm = Swarm(10, target_image, label, model)
    for i in range(1000):
        swarm.optimize()
        if swarm.best_fitness < current_best:
            fig, ax = plt.subplots(1, 2)
            current_best = swarm.best_fitness
            p = swarm.get_best_particle()[0]
            ax[0].imshow(p, cmap='gray')
            ax[1].imshow(target_image, cmap='gray')
            ax[0].set_title('Adversarial')
            ax[1].set_title('Original')
            pred = np.argmax(model.predict(np.expand_dims(p, axis=0)))
            fig.suptitle(f'Iteration {swarm.iteration}, L2-distance: {np.round(swarm.best_fitness, 4)}, Prediction: {pred}')
            fig.tight_layout()

            plt.show()