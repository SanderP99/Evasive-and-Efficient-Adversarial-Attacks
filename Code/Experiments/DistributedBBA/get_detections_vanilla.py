import ast
import csv
import os

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
    bb_model = load_model('../../MNIST/models/cifar', compile=False)
    mnist = CIFAR()
    experiments = pd.read_csv('../experiments_cifar_sorted2.csv', index_col='index')
    dataset = 'cifar'
    output_file = f'results/results_{dataset}_vanilla.csv'
    if not os.path.isfile(output_file):
        with open(output_file, 'w') as file:
            writer = csv.writer(file)
            writer.writerow([
                'original_index', 'y_orig', 'y_target', 'n_particles', 'n_nodes', 'distance',
                'n_detections', 'calls', 'detections_per_node', 'distribution_scheme',
                'source_step', 'spherical_step', 'dataset', 'source_step_multiplier_up',
                'source_step_multiplier_down', 'detections_all'
            ])

    for i in range(2, 20):
        np.random.seed(42)
        experiment = experiments.iloc[i]
        x_orig = mnist.test_data[experiment.name]

        targets = ast.literal_eval(experiment.targets)
        random_inits = mnist.test_data[
            np.array(targets)[np.random.choice(len(targets), size=1, replace=False)]][0]

        attack = BiasedBoundaryAttack(bb_model, create_perlin_noise)
        node = Node(0, dataset, weights_path_mnist=f'../../Defense/{dataset.upper()}encoder.h5')
        adv_example = attack.run_attack(x_orig, experiment.y_target, True, random_inits, (lambda: 25000 - attack.calls),
                                        maximal_calls=25000, dimensions=x_orig.shape, node=node, recalc_mask_every=1000,
                                        source_step=0.002, spherical_step=0.05)

        detections_all = [node.detector.get_detections()]
        total_detections = np.sum([len(x) for x in detections_all])
        distance = np.linalg.norm(adv_example - x_orig)

        with open(output_file, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([
                experiment.name, experiment.y_orig, experiment.y_target, 0, 1,
                distance, total_detections, 25000,
                [len(x) for x in detections_all], 'vanilla', 0.002, 0.05, dataset,
                '', '', detections_all
            ])
