import csv

import numpy as np
import pandas as pd
import ast

from keras.models import load_model
from tqdm import tqdm

from BBA.bba_pso import ParticleBiasedBoundaryAttack
from MNIST.setup_mnist import MNIST


def main():
    np.random.seed(42)
    experiments = pd.read_csv('experiments_sorted.csv', index_col='index')
    n_experiments = 5
    mnist = MNIST()
    bb_model = load_model('../MNIST/models/mnist', compile=False)
    for i in range(2, n_experiments):
        for n_particles in [5, 10]:
            experiment = experiments.iloc[i]
            x_orig = mnist.test_data[experiment.name]
            closest_inits = mnist.test_data[ast.literal_eval(experiment.sorted_targets)[:n_particles]]
            print(closest_inits.shape)
            targets = ast.literal_eval(experiment.targets)
            random_inits = mnist.test_data[
                np.array(targets)[np.random.choice(len(targets), size=n_particles, replace=False)]]
            swarm = ParticleBiasedBoundaryAttack(n_particles=n_particles, model=bb_model,
                                                 target_img=x_orig,
                                                 target_label=experiment.y_target, inits=random_inits)
            n_calls_max = 10000
            filename = f'../BBA/distance_files/distances_pso_{experiment.name}_{experiment.y_target}_{n_particles}_random.csv'
            with open(filename, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(['Queries', 'Distance'])

            with open(filename, 'a') as file:
                writer = csv.writer(file)
                writer.writerow([swarm.total_queries, swarm.best_fitness])

            for _ in tqdm(range(1000)):
                swarm.optimize()
                with open(filename, 'a') as file:
                    writer = csv.writer(file)
                    writer.writerow([swarm.total_queries, swarm.best_fitness])
                # if swarm.total_queries > n_calls_max:
                #     break
            swarm = ParticleBiasedBoundaryAttack(n_particles=n_particles, model=bb_model,
                                                 target_img=x_orig,
                                                 target_label=experiment.y_target, inits=closest_inits)
            current_best = np.infty
            filename = f'../BBA/distance_files/distances_pso_{experiment.name}_{experiment.y_target}_{n_particles}_closest.csv'
            with open(filename, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(['Queries', 'Distance'])

            with open(filename, 'a') as file:
                writer = csv.writer(file)
                writer.writerow([swarm.total_queries, swarm.best_fitness])

            for _ in tqdm(range(1000)):
                swarm.optimize()
                with open(filename, 'a') as file:
                    writer = csv.writer(file)
                    writer.writerow([swarm.total_queries, swarm.best_fitness])
                # if swarm.total_queries > n_calls_max:
                #     break


if __name__ == '__main__':
    main()
