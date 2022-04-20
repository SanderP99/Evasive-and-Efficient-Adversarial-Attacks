import ast
import csv
import os.path

import numpy as np
import pandas as pd
from keras.models import load_model
from tqdm import tqdm

from Attacks.DistributedBBA.distributed_bba import DistributedBiasedBoundaryAttack
from Attacks.DistributedBBA.distribution_scheme import RoundRobinDistribution, DistanceBasedDistributionScheme
from MNIST.setup_cifar import CIFAR
from MNIST.setup_mnist import MNIST

if __name__ == '__main__':
    # Settings
    dataset = 'mnist'  # mnist or cifar
    n_particles = [5]
    n_nodes = [1, 5]
    experiment_ids = list(range(20))
    max_queries = 25000
    distribution_schemes = ['rr']  # rr or mrr or dbl2 or dbe
    source_step_multiplier_up = 1.05
    source_step_multiplier_down = 0.99
    spherical_step = 0.05

    if dataset == 'mnist':
        data = MNIST()
        experiments = pd.read_csv('../experiments_sorted.csv', index_col='index')
        source_step = 0.25
    elif dataset == 'cifar':
        data = CIFAR()
        experiments = pd.read_csv('../experiments_cifar_sorted.csv', index_col='index')
        source_step = 0.20
    else:
        raise ValueError
    bb_model = load_model(f'../../MNIST/models/{dataset}', compile=False)

    for experiment_id in experiment_ids:
        for nodes in n_nodes:
            for particles in n_particles:
                for distribution_scheme in distribution_schemes:
                    np.random.seed(42)

                    output_file = f'results/results_{dataset}_{distribution_scheme}.csv'
                    if not os.path.isfile(output_file):
                        with open(output_file, 'w') as file:
                            writer = csv.writer(file)
                            writer.writerow([
                                'original_index', 'y_orig', 'y_target', 'n_particles', 'n_nodes', 'distance',
                                'n_detections', 'calls', 'detections_per_node', 'distribution_scheme',
                                'source_step', 'spherical_step', 'dataset', 'source_step_multiplier_up',
                                'source_step_multiplier_down'
                            ])

                    use_node_manager = False
                    experiment = experiments.iloc[experiment_id]
                    x_orig = data.test_data[experiment.name]
                    targets = ast.literal_eval(experiment.targets)
                    random_inits = data.test_data[
                        np.array(targets)[np.random.choice(len(targets), size=particles, replace=False)]]

                    if distribution_scheme == 'rr':
                        use_node_manager = True
                        scheme = RoundRobinDistribution(None, n_nodes=nodes, n_particles=particles)
                    elif distribution_scheme == 'mrr':
                        scheme = RoundRobinDistribution(None, n_nodes=nodes, n_particles=particles)
                    elif distribution_scheme == 'dbl2':
                        scheme = DistanceBasedDistributionScheme(None, n_nodes=nodes, dataset=dataset)
                    else:
                        raise ValueError

                    attack = DistributedBiasedBoundaryAttack(n_particles=particles, model=bb_model,
                                                             target_img=x_orig, target_label=experiment.y_target,
                                                             inits=random_inits,
                                                             distribution_scheme=scheme,
                                                             n_nodes=nodes, mapping=None,
                                                             dataset=dataset, spherical_step=spherical_step,
                                                             source_step_multiplier_up=source_step_multiplier_up,
                                                             source_step_multiplier_down=source_step_multiplier_down,
                                                             use_node_manager=use_node_manager,
                                                             source_step=source_step)

                    previous_queries = 0
                    new_queries = 0
                    with tqdm(total=max_queries) as pbar:
                        while attack.swarm.total_queries < max_queries:
                            attack.attack()
                            new_queries, previous_queries = attack.swarm.total_queries, new_queries
                            pbar.update(new_queries - previous_queries)

                    detections_all = [node.detector.get_detections() for node in attack.nodes]
                    total_detections = np.sum([len(x) for x in detections_all])

                    with open(output_file, 'a') as file:
                        writer = csv.writer(file)
                        writer.writerow([
                            experiment.name, experiment.y_orig, experiment.y_target, particles, nodes,
                            attack.swarm.best_fitness, total_detections, attack.swarm.total_queries,
                            [len(x) for x in detections_all], distribution_scheme, source_step, spherical_step, dataset,
                            source_step_multiplier_up, source_step_multiplier_down
                        ])
