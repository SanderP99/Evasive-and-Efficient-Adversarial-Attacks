import ast
import csv
import os
from collections import deque
from typing import Optional, List

import numpy as np
import pandas as pd
from keras.models import Model, load_model
from tqdm import tqdm

from Attacks.DistributedBBA.distributed_bba import DistributedBiasedBoundaryAttack
from Attacks.DistributedBBA.distribution_scheme import CombinationDistributionScheme
from Attacks.DistributedBBA.experiment import Experiment
from Attacks.DistributedBBA.node import Node
from MNIST.setup_mnist import MNIST


class CombinedDistributedBiasedBoundaryAttack:
    def __init__(self, experiments: List[Experiment], n_particles: int, model: Model,
                 distribution_scheme, mapping: Optional[deque] = None,
                 n_nodes: Optional[int] = None, dataset=None, source_step: float = 1e-2,
                 spherical_step: float = 5e-2, steps_per_iteration: int = 50, source_step_multiplier_up: float = 1.05,
                 source_step_multiplier_down: float = 0.6, notify: bool = False):
        self.n_experiments = len(experiments)
        self.attacks = []
        self.nodes = [
            Node(i, dataset, weights_path_mnist=f'../../Defense/{str.upper(dataset)}encoder.h5', notify=notify) for i
            in range(n_nodes)]
        self.node_idx = 0
        self.n_nodes = n_nodes
        self.n_particles = n_particles
        self.total_queries = 0

        for experiment in experiments:
            experiment.set_random_inits(n_particles)
            self.attacks.append(DistributedBiasedBoundaryAttack(n_particles, experiment.random_inits, experiment.x_orig,
                                                                experiment.y_target,
                                                                model, distribution_scheme, mapping, n_nodes, dataset,
                                                                source_step,
                                                                spherical_step, steps_per_iteration,
                                                                source_step_multiplier_up, source_step_multiplier_down,
                                                                True, 'combination', notify=notify))

    def attack(self):
        for a in self.attacks:
            a.attack()

        # Submit the queries
        self.submit_queries()

        # Clear the buffers
        for a in self.attacks:
            for p in a.swarm.particles:
                p.node_manager.clear()

        self.total_queries = sum([a.swarm.total_queries for a in self.attacks])

    def submit_queries(self):
        for i in range(self.n_particles):
            for a in self.attacks:
                p = a.swarm.particles[i]
                for q in p.node_manager.queries:
                    self.nodes[self.node_idx].add_to_detector(q)
                    self.node_idx += 1
                    self.node_idx %= self.n_nodes


if __name__ == '__main__':
    experiments = pd.read_csv('../../Experiments/experiments_sorted.csv', index_col='index')
    data = MNIST()
    model = load_model('../../MNIST/models/mnist', compile=False)
    max_queries = 25000
    exps = []
    for j in range(2):
        e = experiments.iloc[j]
        exps.append(Experiment(e.name, data.test_data[e.name], ast.literal_eval(e.targets), e.y_target, 'mnist'))
    attack = CombinedDistributedBiasedBoundaryAttack(exps, 5, model,
                                                CombinationDistributionScheme(None, n_experiments=len(exps)),
                                                None, 10, 'mnist')

    output_file = f'results/results_{dataset}_{distribution_scheme}.csv'
    if not os.path.isfile(output_file):
        with open(output_file, 'w') as file:
            writer = csv.writer(file)
            writer.writerow([
                'original_index', 'y_orig', 'y_target', 'n_particles', 'n_nodes', 'distance',
                'n_detections', 'calls', 'detections_per_node', 'distribution_scheme',
                'source_step', 'spherical_step', 'dataset', 'source_step_multiplier_up',
                'source_step_multiplier_down', 'detections_all', 'threshold'
            ])

    previous_queries = 0
    new_queries = 0
    with tqdm(total=len(exps) * max_queries) as pbar:
        while attack.total_queries < max_queries:
            attack.attack()
            new_queries, previous_queries = attack.total_queries, new_queries
            pbar.update(new_queries - previous_queries)

    detections_all = [node.detector.get_detections() for node in attack.nodes]
    total_detections = np.sum([len(x) for x in detections_all])

    with open(output_file, 'a') as file:
        writer = csv.writer(file)
        writer.writerow([
            experiment.name, experiment.y_orig, experiment.y_target, particles, nodes,
            attack.swarm.best_fitness, total_detections, attack.swarm.total_queries,
            [len(x) for x in detections_all], str(scheme), source_step, spherical_step, dataset,
            source_step_multiplier_up, source_step_multiplier_down, detections_all, threshold
        ])
