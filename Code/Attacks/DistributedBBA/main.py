from collections import deque

import numpy as np
from keras.models import load_model
from tqdm import tqdm

from Attacks.DistributedBBA.distributed_bba import DistributedBiasedBoundaryAttack
from Attacks.DistributedBBA.distribution_scheme import RoundRobinDistribution
from MNIST.setup_mnist import MNIST

if __name__ == '__main__':
    bb_model = load_model('../../MNIST/models/mnist', compile=False)
    mnist = MNIST()
    n_particles = 1
    target_label = 3
    image_index = 1302
    labels = np.argmax(mnist.test_labels, axis=1)
    labels = mnist.test_data[labels == target_label]
    inits = labels[:n_particles]
    n_nodes = 1
    mapping = deque([0] * n_particles)
    attack = DistributedBiasedBoundaryAttack(n_particles=n_particles, model=bb_model,
                                             target_img=mnist.test_data[image_index],
                                             target_label=target_label, inits=inits,
                                             distribution_scheme=RoundRobinDistribution(),
                                             n_nodes=n_nodes, mapping=mapping)

    for i in tqdm(range(12)):
        attack.attack()
    detections_all = [node.detector.get_detections() for node in attack.nodes]
    print("Best fitness value: ", attack.swarm.best_fitness)
    print("Total detections: ", np.sum([len(x) for x in detections_all]))
    print("\n")
    sum = 0

    for j in range(n_nodes):
        detections = detections_all[j]
        sum += attack.nodes[j].detector.num_queries
        print("Num detections:", len(detections))
        print("Queries per detection:", detections)
        print("i-th query that caused detection:", attack.nodes[j].detector.history)
        print("\n")
