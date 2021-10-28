import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

from pytorch.train_adversarial_mnist import LeNet5

with open('clipped.txt', 'rb') as f:
    clipped = pickle.load(f)

plt.imshow(np.array(clipped[0]).reshape((28,28,1)))
plt.show()

model = LeNet5()
model.load_state_dict(torch.load('../pytorch/mnist_cnn.pt'))
model.eval()

print(torch.argmax(model(clipped[0][None, ...])))