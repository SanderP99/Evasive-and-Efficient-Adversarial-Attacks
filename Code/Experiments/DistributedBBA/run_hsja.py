import ast
import csv
import os

import numpy as np
import pandas as pd
from keras.models import load_model

from Attacks.DistributedBBA.distribution_scheme import RoundRobinDistribution
from Attacks.HSJA.hsja_vanilla import hsja
from MNIST.setup_cifar import CIFAR
from MNIST.setup_mnist import MNIST

if __name__ == '__main__':
    datasets = ['mnist', 'cifar']
    experiment_ids = list(range(20))
    max_queries = 25000
    n_nodes = [10]

    for dataset in datasets:
        if dataset == 'mnist':
            data = MNIST()
            experiments = pd.read_csv('../experiments_sorted.csv', index_col='index')
        elif dataset == 'cifar':
            data = CIFAR()
            experiments = pd.read_csv('../experiments_cifar_sorted2.csv', index_col='index')
        else:
            raise ValueError
        bb_model = load_model(f'../../MNIST/models/{dataset}', compile=False)

        for experiment_id in experiment_ids:
            print(experiment_id)
            for nodes in n_nodes:
                np.random.seed(42)
                output_file = f'results/results_{dataset}_hsja.csv'
                if not os.path.isfile(output_file):
                    with open(output_file, 'w') as file:
                        writer = csv.writer(file)
                        writer.writerow([
                            'original_index', 'y_orig', 'y_target', 'n_nodes', 'distance',
                            'n_detections', 'calls', 'detections_per_node', 'distribution_scheme',
                            'dataset', 'detections_all'
                        ])

                experiment = experiments.iloc[experiment_id]
                x_orig = data.test_data[experiment.name]
                targets = ast.literal_eval(experiment.targets)
                random_inits = data.test_data[
                    np.array(targets)[np.random.choice(len(targets), size=1, replace=False)]]
                use_node_manager = True
                scheme = RoundRobinDistribution(None, n_nodes=nodes, n_particles=1)

                example, qdw = hsja(bb_model, np.expand_dims(x_orig, axis=0), target_label=experiment.y_target,
                                    target_image=random_inits[0], num_iterations=10, distributed=True,
                                    flush_buffer_after_detection=True, n_nodes=nodes, max_queries=max_queries,
                                    verbose=False, dataset=dataset)

                detections_all = [node.detector.get_detections() for node in qdw.nodes]

                with open(output_file, 'a') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        experiment.name, experiment.y_orig, experiment.y_target, nodes,
                        np.linalg.norm(example - x_orig), qdw.get_n_detections(), max_queries,
                        [len(x) for x in detections_all], str(scheme), dataset,
                        detections_all
                    ])

