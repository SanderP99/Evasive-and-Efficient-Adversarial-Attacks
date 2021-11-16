import numpy as np
from keras.models import load_model

from Attacks.DistributedBBA.distributed_bba import DistributedBiasedBoundaryAttack
from Attacks.DistributedBBA.distribution_scheme import RoundRobinDistribution
from MNIST.setup_mnist import MNIST

if __name__ == '__main__':
    bb_model = load_model('../../MNIST/models/mnist', compile=False)
    mnist = MNIST()
    n_particles = 5
    target_label = 3
    image_index = 1302
    labels = np.argmax(mnist.test_labels, axis=1)
    labels = mnist.test_data[labels == target_label]
    inits = labels[:n_particles]
    attack = DistributedBiasedBoundaryAttack(n_particles=n_particles, model=bb_model,
                                             target_img=mnist.test_data[image_index],
                                             target_label=target_label, inits=inits,
                                             distribution_scheme=RoundRobinDistribution())
    for i in range(10):
        print(attack.mapping)
        attack.attack()
        print([np.array(node.detector.buffer).shape for node in attack.nodes])
