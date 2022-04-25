import ast
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def write_to_file(detections, file):
    with open(file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['detections', 'calls', 'ci'])
        y = np.mean(detections, axis=0)
        ci = 1.96 * np.std(detections, axis=0) / np.sqrt(20)
        previous = 0, 0
        for call, (detection, c) in enumerate(zip(y, ci)):
            if (detection, c) != previous:
                writer.writerow([detection, call, c])
                previous = detection, c


def compare_detections():
    dataset = 'mnist'
    df = pd.read_csv(f'../Experiments/DistributedBBA/results/results_{dataset}_vanilla.csv')

    detections = fill_detections_array(df)
    write_to_file(detections, f'detections_bba_{dataset}.csv')

    fig, ax = plt.subplots()
    y = np.mean(detections, axis=0)
    ax.plot(y)
    ci = 1.96 * np.std(detections, axis=0) / np.sqrt(20)
    ax.fill_between(range(25000), (y - ci), (y + ci), color='b', alpha=.1)

    df = pd.read_csv(f'../Experiments/DistributedBBA/results/results_{dataset}_rr.csv')
    df = df[df.n_nodes == 1]
    df = df[df.n_particles == 1]
    detections = fill_detections_array(df)
    write_to_file(detections, f'detections_bba_pso_{dataset}.csv')

    y = np.mean(detections, axis=0)
    ax.plot(y)
    ci = 1.96 * np.std(detections, axis=0) / np.sqrt(20)
    ax.fill_between(range(25000), (y - ci), (y + ci), color='r', alpha=.1)

    plt.show()


def fill_detections_array(df):
    detections = np.zeros((20, 25000), dtype=int)
    for i in range(20):
        ex = df.iloc[i]
        lst = ast.literal_eval(ex.detections_all)[0]
        indexes = np.cumsum(lst)
        previous_index = 0
        for n_detections, index in enumerate(indexes):
            for j in range(previous_index, index):
                detections[i][j] = n_detections
            previous_index = index
        for j in range(previous_index, 25000):
            detections[i][j] = n_detections + 1
    return detections


if __name__ == '__main__':
    compare_detections()
