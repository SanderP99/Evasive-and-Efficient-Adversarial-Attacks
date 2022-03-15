import ast

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
    max_queries = 100

    # Values
    n_particles = trial.suggest_int('n_particles', 1, 20)
    n_nodes = trial.suggest_int('n_nodes', 1, 10)
    distribution_scheme = RoundRobinDistribution(None)
    total_detections = 0
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
                                                 n_nodes=n_nodes, mapping=None, dataset='mnist')

        previous_queries = 0
        new_queries = 0
        with tqdm(total=max_queries) as pbar:
            while attack.swarm.total_queries < max_queries:
                attack.attack()
                new_queries, previous_queries = attack.swarm.total_queries, new_queries
                pbar.update(new_queries - previous_queries)

        detections_all = [node.detector.get_detections() for node in attack.nodes]
        total_detections += np.sum([len(x) for x in detections_all])
    return total_detections


if __name__ == '__main__':
    study = optuna.create_study(direction="minimize")
    study.optimize(main, n_trials=3)
