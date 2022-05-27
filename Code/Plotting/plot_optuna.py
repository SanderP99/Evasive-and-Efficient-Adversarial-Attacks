import csv
import pickle

import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.visualization.matplotlib import plot_contour


def plot_optuna(dataset):
    study = optuna.load_study(f'{dataset}_optuna_log',
                              f'sqlite:///../Experiments/DistributedBBA/results/{dataset}_optuna_log_log.db')
    fig = plot_contour(study, target=distance, target_name='Distance')
    plt.show()
    # with open('fig.pkl', 'wb') as file:
    #     pickle.dump(fig, file)
    return fig


def n_detections(x):
    return x.values[0]


def distance(x):
    return x.values[1]


def fig_to_data(fig):
    parameter_list = ['c1', 'c2', 'n_particles', 'source_step', 'w_start']
    scatter_is_odd = True
    plot_number = 0
    col = 0
    for i, ax in enumerate(fig['data']):
        if i in [0, 11, 22, 33, 44]:
            # On diagonal
            scatter_is_odd = not scatter_is_odd
            col += 1
        else:
            # Not on diagonal
            if scatter_is_odd == (i % 2 == 1):
                # Scatter plot
                pass
            else:
                # Contour plot
                x_label, y_label = parameter_list[col], parameter_list[plot_number // 9]
                col += 1
                col %= 5
                nb_trials = np.count_nonzero(~np.isnan(ax['z']))
                arr = np.zeros((nb_trials, 3))
                idx = 0
                # print(ax)
                with open(f'../Analysis/contour_files/contour_{x_label}_{y_label}.csv', 'w') as file:
                    writer = csv.writer(file)
                    for row_i, row_v in enumerate(ax['z']):
                        writer.writerow([])
                        for col_i, col_v in enumerate(row_v):
                            # if not np.isnan(col_v):
                            y = ax['x'][col_i]
                            x = ax['y'][row_i]
                            z = col_v
                            writer.writerow([x, y, z])
                            idx += 1

                # np.savetxt(f'../Analysis/contour_files/contour_{x_label}_{y_label}.csv', arr, delimiter=',')

        plot_number += 1


if __name__ == '__main__':
    dataset = 'cifar'
    f = plot_optuna(dataset)
    # fig_to_data(f)
    # with open('fig.pkl', 'rb') as file:
    #     fig = pickle.load(file)
    # fig.get_xdata()

