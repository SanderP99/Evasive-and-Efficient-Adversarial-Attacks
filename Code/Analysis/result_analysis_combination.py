from os import listdir

import numpy as np
import pandas as pd


def analyse_results(path):
    dfs = []
    for file in listdir(path):
        if str.__contains__(file, 'comb'):
            x = pd.read_csv(path + file)
            x.columns = [name.strip() for name in x.columns]
            dfs.append(x)
    df = pd.concat(dfs, axis=0)


    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print(df.groupby(['dataset', 'distribution_scheme', 'n_particles', 'n_nodes']).agg({
                                                                                        'n_detections':
                                                                                            [np.mean,
                                                                                             np.std]
                                                                                        }))


if __name__ == '__main__':
    path = '../Experiments/DistributedBBA/results/'
    analyse_results(path)
