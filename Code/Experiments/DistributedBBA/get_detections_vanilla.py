import ast
import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import load_model

from Attacks.DistributedBBA.node import Node
from Attacks.TargetedBBA.bba import BiasedBoundaryAttack
from Attacks.TargetedBBA.sampling_provider import create_perlin_noise
from MNIST.setup_cifar import CIFAR
from MNIST.setup_mnist import MNIST

if __name__ == '__main__':
    bb_model = load_model('../MNIST/models/mnist', compile=False)
    mnist = MNIST()
    experiments = pd.read_csv('experiments.csv', index_col='index')

    output_file = 'vanilla_detections.csv'

    for i in [1]:
        np.random.seed(42)
        experiment = experiments.iloc[i]
        x_orig = mnist.test_data[experiment.name]

        targets = ast.literal_eval(experiment.targets)
        random_inits = mnist.test_data[
            np.array(targets)[np.random.choice(len(targets), size=1, replace=False)]][0]
        plt.imshow(random_inits)
        plt.show()
        plt.imshow(x_orig)
        plt.show()

        attack = BiasedBoundaryAttack(bb_model, create_perlin_noise)
        node = Node(0, 'cifar', weights_path_mnist='../Defense/MNISTencoder.h5')
        adv_example = attack.run_attack(x_orig, experiment.y_target, True, random_inits, (lambda: 25000 - attack.calls),
                                        maximal_calls=25000, dimensions=x_orig.shape, node=node, recalc_mask_every=1000,
                                        source_step=0.002, spherical_step=0.05)

        detections_all = [node.detector.get_detections()]
        print(detections_all)
        total_detections = np.sum([len(x) for x in detections_all])
        print('Total: ', total_detections)
        distance = np.linalg.norm(adv_example - x_orig)
        print('Distance: ', distance)

        with open(output_file, 'a') as file:
            writer = csv.writer(file)
            writer.writerow(
                [experiment.name, experiment.y_orig, experiment.y_target, total_detections, distance, attack.calls,
                 'cifar'])
