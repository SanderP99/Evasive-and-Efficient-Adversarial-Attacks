import ast
import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import load_model

from Attacks.HSJA.hsja_vanilla import hsja
from MNIST.setup_mnist import MNIST

if __name__ == '__main__':
    model = load_model('../../MNIST/models/mnist', compile=False)
    mnist = MNIST()
    experiments = pd.read_csv('../../Experiments/experiments_sorted.csv', index_col='index')

    for idx in range(4, 5):
        for n_nodes in [1, 5, 10]:
            for flush_buffer in [True, False]:
                print(idx)
                experiment = experiments.iloc[idx]
                x_orig = mnist.test_data[experiment.name]
                targets = ast.literal_eval(experiment.targets)
                random_inits = mnist.test_data[
                    np.array(targets)[np.random.choice(len(targets), size=1, replace=False)]]

                np.random.seed(42)
                example, qdw = hsja(model, np.expand_dims(x_orig, axis=0), target_label=experiment.y_target,
                                    target_image=random_inits[0], num_iterations=100, distributed=True,
                                    flush_buffer_after_detection=flush_buffer, n_nodes=n_nodes)

                with open('../../Experiments/DistributedBBA/hsja_mnist.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([experiment.name, experiment.y_orig, experiment.y_target, qdw.n_queries, n_nodes,
                                     np.linalg.norm(example - x_orig), qdw.get_n_detections(),
                                     1 if flush_buffer else 0])

                print(np.argmax(model.predict(example)), np.linalg.norm(example - x_orig))
                print(qdw.n_queries, qdw.get_n_detections())
