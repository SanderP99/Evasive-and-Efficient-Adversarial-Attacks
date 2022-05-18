import ast
from collections import defaultdict
from os import listdir

import numpy as np
import pandas as pd


def compare(path):
    dfs_comb = []
    dfs_rest = []
    for file in listdir(path):
        if str.__contains__(file, 'comb'):
            x = pd.read_csv(path + file)
            x.columns = [name.strip() for name in x.columns]
            dfs_comb.append(x)
        else:
            x = pd.read_csv(path + file)
            x.columns = [name.strip() for name in x.columns]
            dfs_rest.append(x)
    df_comb = pd.concat(dfs_comb, axis=0)
    df_rest = pd.concat(dfs_rest, axis=0)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    d = defaultdict()
    for i, row in df_comb.iterrows():
        indices = ast.literal_eval(row.original_indices)
        distances = ast.literal_eval(row.distances)
        dataset = row.dataset
        n_particles = row.n_particles
        n_nodes = row.n_nodes
        df_filtered = df_rest[
            (df_rest.n_nodes == n_nodes) & (df_rest.n_particles == n_particles) & (df_rest.dataset == dataset) & (
                df_rest.original_index.isin(indices)) & (
                    df_rest.distribution_scheme.apply(str.strip) == 'round_robin')]
        assert len(indices) == len(df_filtered)
        d[str(indices)] = [row.n_detections, df_filtered['n_detections'].sum(), sum(distances),
                           df_filtered['distance'].sum(), len(indices)]

    df = pd.DataFrame.from_dict(d, orient='index',
                                columns=['n_detections_combined', 'n_detection_split', 'distance_combined',
                                         'distance_split', 'n_experiments'])
    df.to_csv('combination_comparison.csv')


if __name__ == '__main__':
    path = '../Experiments/DistributedBBA/results/'
    compare(path)
