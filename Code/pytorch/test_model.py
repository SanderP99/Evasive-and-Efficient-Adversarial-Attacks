import numpy as np
import torch
from torch.autograd import Variable

from MNIST.setup_mnist import MNIST
from pytorch.train_adversarial_mnist import LeNet5

if __name__ == '__main__':
    # Load the model
    model = LeNet5()
    model.load_state_dict(torch.load('mnist_cnn.pt'))
    model.eval()

    # Load the data
    mnist = MNIST()
    total = 0
    correct = 0
    for img, label in zip(mnist.test_data, mnist.test_labels):
        total += 1
        img = Variable(torch.from_numpy(img), requires_grad=False, volatile=False)[None, ...]
        img = np.transpose(img, axes=(0, 3, 1, 2))
        scores = model(img)
        if scores.data.cpu().max(1)[1] == np.argmax(label):
            correct += 1

    print(correct / total)
