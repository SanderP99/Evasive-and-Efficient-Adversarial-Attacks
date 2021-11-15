import numpy as np
import eagerpy as ep
import torch
from foolbox.attacks import BoundaryAttack
from foolbox import TensorFlowModel, PyTorchModel
from keras.models import load_model
import pickle

from matplotlib import pyplot as plt
from torch.autograd import Variable

from MNIST.setup_mnist import MNIST
from pytorch.train_adversarial_mnist import LeNet5

if __name__ == '__main__':
    model = LeNet5()
    model.load_state_dict(torch.load('../pytorch/mnist_cnn.pt'))
    model = model.eval()
    fmodel = PyTorchModel(model, bounds=(0, 1))

    mnist = MNIST()
    index = 500
    amount = 1
    # img = mnist.test_data[index]
    # target_image = Variable(torch.from_numpy(img), requires_grad=False, volatile=False)[None, ...]
    # img = np.transpose(target_image, axes=(0, 3, 1, 2))
    target_images = torch.from_numpy(np.transpose(mnist.test_data[index:index+amount], axes=(0, 3, 1, 2)))
    # label = torch.argmax(torch.from_numpy(mnist.test_labels[index]))[None, ...]
    # print(label)
    # label = torch.tensor([5])

    labels = torch.from_numpy(np.argmax(mnist.test_labels[index:index+amount], axis=1))
    plt.imshow(mnist.test_data[index:index+1].reshape((28, 28, 1)))
    plt.show()
    # target_images, labels = ep.astensors(target_images, labels)
    print(model(target_images), labels)
    attack = BoundaryAttack(steps=1000000)
    epsilons = [
        # 0.0,
        # 0.0002,
        # 0.0005,
        # 0.0008,
        # 0.001,
        # 0.0015,
        # 0.002,
        # 0.003,
        0.01,
        0.1,
        0.3,
        0.5,
        1.0,
    ]
    # Epsilon values are used to clip the perturbation, but this is only taken into account at the end and
    # is therefore not useful
    raw_advs, clipped_advs, success = attack(fmodel, target_images, labels, epsilons=None)

    # # raw_advs, clipped_advs, success = attack(fmodel, target_images, labels, epsilons=epsilons)
    f = open('UntargetedBBA/raw.txt', 'wb')
    pickle.dump(raw_advs, f)
    f.close()
    with open('UntargetedBBA/clipped.txt', 'wb') as f:
        pickle.dump(clipped_advs, f)

    plt.imshow(np.array(clipped_advs[0]).reshape((28, 28, 1)))
    plt.show()
    print(success)

