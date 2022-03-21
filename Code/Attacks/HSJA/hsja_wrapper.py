import ast

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

    experiment = experiments.iloc[0]
    x_orig = mnist.test_data[experiment.name]
    targets = ast.literal_eval(experiment.targets)
    random_inits = mnist.test_data[
        np.array(targets)[np.random.choice(len(targets), size=1, replace=False)]]

    print(experiment.y_target)
    plt.imshow(x_orig)
    plt.show()

    example = hsja(model, np.expand_dims(x_orig, axis=0), target_label=experiment.y_target,
                   target_image=random_inits[0])

    plt.imshow(example[0])
    plt.show()
