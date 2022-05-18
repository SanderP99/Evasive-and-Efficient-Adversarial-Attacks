import ast
import csv
import os

import pandas as pd
from flatbuffers.builder import np
from keras.models import load_model
from tqdm import tqdm

from Attacks.DistributedBBA.combined_distributed_bba import CombinedDistributedBiasedBoundaryAttack
from Attacks.DistributedBBA.distribution_scheme import CombinationDistributionScheme
from Attacks.DistributedBBA.experiment import Experiment
from MNIST.setup_cifar import CIFAR
from MNIST.setup_mnist import MNIST

if __name__ == '__main__':
    # Settings
    dataset = 'mnist'  # mnist or cifar
    n_particles = [5]
    n_nodes = [10]
    experiment_ids = list(range(20))
    max_queries = 25000
    n_experiments = 3
    distribution_schemes = ['comb']
    source_step_multiplier_up = 1.05
    source_step_multiplier_down = 0.99
    spherical_step = 0.05

    if dataset == 'mnist':
        data = MNIST()
        experiments = pd.read_csv('../experiments_sorted.csv', index_col='index')
        source_step = 0.25
    elif dataset == 'cifar':
        data = CIFAR()
        experiments = pd.read_csv('../experiments_cifar_sorted2.csv', index_col='index')
        source_step = 0.20
    else:
        raise ValueError
    bb_model = load_model(f'../../MNIST/models/{dataset}', compile=False)

    for experiment_id in experiment_ids[:-n_experiments + 1]:
        for nodes in n_nodes:
            for particles in n_particles:
                for distribution_scheme in distribution_schemes:
                    np.random.seed(42)
                    output_file = f'results/results_{dataset}_{distribution_scheme}.csv'
                    if not os.path.isfile(output_file):
                        with open(output_file, 'w') as file:
                            writer = csv.writer(file)
                            writer.writerow([
                                'original_indices', 'y_origs', 'y_targets', 'n_particles', 'n_nodes', 'distances',
                                'n_detections', 'calls', 'detections_per_node', 'distribution_scheme',
                                'source_step', 'spherical_step', 'dataset', 'source_step_multiplier_up',
                                'source_step_multiplier_down', 'detections_all'
                            ])

                    use_node_manager = True
                    distance_based = 'combination'
                    scheme = CombinationDistributionScheme(None, n_experiments=n_experiments)

                    exps = []
                    for j in range(n_experiments):
                        e = experiments.iloc[experiment_id + j]
                        exps.append(
                            Experiment(e.name, data.test_data[e.name], ast.literal_eval(e.targets), e.y_target,
                                       e.y_orig, dataset))

                    attack = CombinedDistributedBiasedBoundaryAttack(exps, particles, bb_model,
                                                                     scheme, None, nodes, dataset)

                    previous_queries = 0
                    new_queries = 0
                    with tqdm(total=max_queries * n_experiments) as pbar:
                        while attack.total_queries < n_experiments * max_queries:
                            attack.attack()
                            new_queries, previous_queries = attack.total_queries, new_queries
                            pbar.update(new_queries - previous_queries)

                    detections_all = [node.detector.get_detections() for node in attack.nodes]
                    total_detections = np.sum([len(x) for x in detections_all])

                    with open(output_file, 'a') as file:
                        writer = csv.writer(file)
                        writer.writerow([
                            [e.id for e in exps], [e.y_orig for e in exps], [e.y_target for e in exps], particles,
                            nodes,
                            [a.swarm.best_fitness for a in attack.attacks], total_detections, attack.total_queries,
                            [len(x) for x in detections_all], str(scheme), source_step, spherical_step, dataset,
                            source_step_multiplier_up, source_step_multiplier_down, detections_all
                        ])
