import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt

import BAPSO.untargeted_particle
from MNIST.setup_mnist import MNIST

if __name__ == '__main__':
    mnist = MNIST()
    model = load_model('../MNIST/models/mnist', compile=False)
    target_image = mnist.test_data[1]
    label = np.argmax(model.predict(target_image.reshape(1, 28, 28, 1)))

    particle = BAPSO.untargeted_particle.UntargetedParticle(0, target_image, label, model)

    for i in range(500):
        print(i)
        particle.update_velocity(0)
        particle.update_position()
        particle.update_best_fitness()
        if i % 10 == 0:
            plt.imshow(particle.position.reshape((28, 28, 1)), cmap=plt.get_cmap('gray'))
            plt.show()
            print("Prediction: ", np.argmax(model.predict(particle.position.reshape(1, 28, 28, 1))))
