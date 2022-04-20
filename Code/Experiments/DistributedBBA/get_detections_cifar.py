import ast
import csv
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import load_model
from tqdm import tqdm

from Attacks.DistributedBBA.distributed_bba import DistributedBiasedBoundaryAttack
from Attacks.DistributedBBA.distribution_scheme import RoundRobinDistribution, DistanceBasedDistributionScheme
from MNIST.setup_cifar import CIFAR

if __name__ == '__main__':
    bb_model = load_model('../../MNIST/models/cifar', compile=False)
    cifar = CIFAR()
    experiments = pd.read_csv('../experiments_cifar_sorted.csv', index_col='index')

    output_file = 'detections_distributed_vs_non_distributed.csv'
    # with open(output_file, 'w') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(
    #         ['original_index', 'y_orig', 'y_target', 'n_particles', 'n_nodes', 'best_distance', 'n_detections',
    #          'n_calls',
    #          'detections_per_node', 'distribution_scheme'])

    n_experiments = 5
    n_particles = 5
    max_queries = 25000
    for i in [0, 1, 3, 4, 6]:
        for n_nodes in [5, 10]:
            np.random.seed(42)
            if n_nodes == 1:
                mapping = deque([0] * n_particles)
            elif n_nodes == 5:
                mapping = deque(range(n_particles))
            else:
                mapping = None

            experiment = experiments.iloc[i]
            x_orig = cifar.test_data[experiment.name]

            targets = ast.literal_eval(experiment.targets)
            random_inits = cifar.test_data[
                np.array(targets)[np.random.choice(len(targets), size=n_particles, replace=False)]]
            distribution_scheme = RoundRobinDistribution(mapping, n_nodes=n_nodes, n_particles=n_particles)
            # distribution_scheme = DistanceBasedDistributionScheme(mapping, n_nodes, dataset='cifar')
            attack = DistributedBiasedBoundaryAttack(n_particles=n_particles, model=bb_model,
                                                     target_img=x_orig,
                                                     target_label=experiment.y_target, inits=random_inits,
                                                     distribution_scheme=distribution_scheme,
                                                     n_nodes=n_nodes, mapping=mapping, dataset='cifar',
                                                     source_step_multiplier_up=1.05, source_step_multiplier_down=0.99,
                                                     spherical_step=0.05, source_step=0.2,
                                                     use_node_manager=True)
            previous_queries = 0
            new_queries = 0
            with tqdm(total=max_queries) as pbar:
                while attack.swarm.total_queries < max_queries:
                    attack.attack()
                    new_queries, previous_queries = attack.swarm.total_queries, new_queries
                    pbar.update(new_queries - previous_queries)

            # for _ in tqdm(range(1000)):
            #     attack.attack()

            detections_all = [node.detector.get_detections() for node in attack.nodes]
            total_detections = np.sum([len(x) for x in detections_all])

            with open(output_file, 'a') as file:
                writer = csv.writer(file)
                writer.writerow(
                    [experiment.name, experiment.y_orig, experiment.y_target, n_particles, n_nodes,
                     attack.swarm.best_fitness,
                     total_detections, attack.swarm.total_queries, [len(x) for x in detections_all],
                     str(distribution_scheme), 'cifar'])
