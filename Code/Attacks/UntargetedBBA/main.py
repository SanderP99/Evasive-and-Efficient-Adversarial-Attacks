import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
import csv

from tqdm import tqdm
from MNIST.setup_mnist import MNIST
from swarm import Swarm

if __name__ == '__main__':
    mnist = MNIST()
    model = load_model('../../MNIST/models/mnist', compile=False)
    target_image = mnist.test_data[500]
    print(np.min(target_image))
    label = np.argmax(model.predict(target_image.reshape(1, 28, 28, 1)))
    current_best = np.infty
    swarm = Swarm(20, target_image, label, model)
    with open('difference10000.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration', 'Difference'])

    for i in tqdm(range(10000)):
        swarm.optimize()
        with open('difference10000.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow([i, np.linalg.norm(swarm.get_best_particle()[0] - swarm.get_worst_article()[0])])
        if swarm.best_fitness < current_best:
            fig, ax = plt.subplots(1, 3)
            current_best = swarm.best_fitness
            p = swarm.get_best_particle()[0]
            q = swarm.get_worst_article()[0]
            ax[0].imshow(p, cmap='gray')
            ax[1].imshow(target_image, cmap='gray')
            ax[2].imshow(q, cmap='gray')
            ax[0].set_title('Adversarial')
            ax[1].set_title('Original')
            ax[2].set_title('Worst')
            pred = np.argmax(model.predict(np.expand_dims(p, axis=0)))
            fig.suptitle(
                f'Iteration {swarm.iteration}, L2-distance: {np.round(swarm.best_fitness, 4)}, Prediction: {pred}, Total queries: {swarm.total_queries}')
            fig.tight_layout()
            # TODO: plot prediction for worst and compare with vanilla BA

            plt.show()
