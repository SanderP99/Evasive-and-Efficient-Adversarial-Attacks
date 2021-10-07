from MNIST.setup_mnist import MNIST
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    mnist = MNIST()
    model = load_model('models/mnist', compile=False)

    index = 10
    test_images = mnist.test_data[:index]
    test_label = mnist.test_labels[:index]

    # plt.imshow(test_images, cmap=plt.get_cmap('gray'))
    # plt.show()

    results = model.predict(test_images.reshape(index, 28, 28, 1))
    results = np.argmax(results, axis=1)

    for i in range(9):
        ax = plt.subplot(331 + i)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.imshow(test_images[i], cmap=plt.get_cmap('gray'))
        ax.set_title("Pred: " + str(results[i]) + ", Real: " + str(np.argmax(test_label[i])))
    plt.tight_layout()
    plt.show()