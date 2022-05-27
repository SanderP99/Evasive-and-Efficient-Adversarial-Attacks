import ast
import logging
import sys

import numpy as np
import optuna

import pandas as pd
from keras.models import load_model
from tqdm import tqdm

from Attacks.DistributedBBA.distributed_bba import DistributedBiasedBoundaryAttack
from Attacks.DistributedBBA.distribution_scheme import RoundRobinDistribution
from MNIST.setup_cifar import CIFAR
from MNIST.setup_mnist import MNIST

dataset = 'cifar'


def main(trial):
    max_queries = 25000
    n_nodes = 10
    bb_model = load_model(f'../../MNIST/models/{dataset}', compile=False)
    n_particles = trial.suggest_int('n_particles', 1, 20)
    source_step = trial.suggest_float('source_step', 1e-6, 1., log=True)
    w_start = trial.suggest_float('w_start', 0.5, 2.)
    c1 = trial.suggest_float('c1', 0.5, 3.)
    c2 = trial.suggest_float('c2', 0.5, 3.)

    if dataset == 'mnist':
        data = MNIST()
        experiments = pd.read_csv('../experiments_sorted.csv', index_col='index')
    elif dataset == 'cifar':
        data = CIFAR()
        experiments = pd.read_csv('../experiments_cifar_sorted2.csv', index_col='index')
    else:
        raise ValueError

    total_detections = 0
    total_distance = 0
    for i in range(2):
        np.random.seed(42)
        experiment = experiments.iloc[i]
        x_orig = data.test_data[experiment.name]

        targets = ast.literal_eval(experiment.targets)
        random_inits = data.test_data[
            np.array(targets)[np.random.choice(len(targets), size=n_particles, replace=False)]]
        distribution_scheme = RoundRobinDistribution(None, n_nodes=n_nodes, n_particles=n_particles)
        attack = DistributedBiasedBoundaryAttack(n_particles=n_particles, model=bb_model,
                                                 target_img=x_orig,
                                                 target_label=experiment.y_target, inits=random_inits,
                                                 distribution_scheme=distribution_scheme,
                                                 n_nodes=n_nodes, mapping=None, dataset=dataset,
                                                 source_step=source_step,
                                                 spherical_step=0.05, w_start=w_start,
                                                 c1=c1, c2=c2)

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

    study_name = f'{dataset}_optuna_log'  # Unique identifier of the study.
    storage_name = "sqlite:///results/{}_log.db".format(study_name)
    study = optuna.create_study(directions=["minimize", "minimize"], study_name=study_name, storage=storage_name,
                                load_if_exists=True)
    study.optimize(main, n_trials=20)
