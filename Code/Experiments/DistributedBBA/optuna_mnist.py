import ast
import logging
import sys

import numpy as np
import optuna
import pandas as pd
from keras.models import load_model
from tqdm import tqdm

from Attacks.DistributedBBA.distributed_bba import DistributedBiasedBoundaryAttack
from Attacks.DistributedBBA.distribution_scheme import DistanceBasedDistributionScheme, RoundRobinDistribution
from MNIST.setup_mnist import MNIST


def main(trial):
    bb_model = load_model('../MNIST/models/mnist', compile=False)
    mnist = MNIST()
    experiments = pd.read_csv('experiments_sorted.csv', index_col='index')
    max_queries = 25000

    # Values
    n_particles = trial.suggest_int('n_particles', 1, 20)
    n_nodes = trial.suggest_int('n_nodes', 1, 10)
    source_step = trial.suggest_float('source_step', 1e-6, 1e-1)
    steps_per_iteration = trial.suggest_int('steps_per_iteration', 10, 51)
    spherical_step = trial.suggest_float('spherical_step', 1e-7, 5e-2)
    distribution_scheme = RoundRobinDistribution(None, n_nodes=n_nodes, n_particles=n_particles)

    total_detections = 0
    total_distance = 0
    for i in range(3):
        experiment = experiments.iloc[i]
        x_orig = mnist.test_data[experiment.name]

        targets = ast.literal_eval(experiment.targets)
        random_inits = mnist.test_data[
            np.array(targets)[np.random.choice(len(targets), size=n_particles, replace=False)]]

        attack = DistributedBiasedBoundaryAttack(n_particles=n_particles, model=bb_model,
                                                 target_img=x_orig,
                                                 target_label=experiment.y_target, inits=random_inits,
                                                 distribution_scheme=distribution_scheme,
                                                 n_nodes=n_nodes, mapping=None, dataset='mnist',
                                                 source_step=source_step,
                                                 steps_per_iteration=steps_per_iteration, spherical_step=spherical_step)

        previous_queries = 0
        new_queries = 0
        with tqdm(total=max_queries) as pbar:
            while attack.swarm.total_queries < max_queries:
                attack.attack()
                new_queries, previous_queries = attack.swarm.total_queries, new_queries
                pbar.update(new_queries - previous_queries)

        detections_all = [node.detector.get_detections() for node in attack.nodes]
        total_detections += np.sum([len(x) for x in detections_all])
        total_distance += attack.swarm.best_fitness
    return total_detections, total_distance


if __name__ == '__main__':
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "mnist"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(directions=["minimize", "minimize"], study_name=study_name, storage=storage_name,
                                load_if_exists=True)
    study.optimize(main, n_trials=20)
