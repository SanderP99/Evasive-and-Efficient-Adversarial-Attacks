from os import listdir

import pandas as pd


def analyse_results(path: str) -> None:
    dfs = []
    for file in listdir(path):
        x = pd.read_csv(path + file)
        x.columns = [name.strip() for name in x.columns]
        dfs.append(x)
    df = pd.concat(dfs, axis=0)
    print(df.groupby(['dataset', 'distribution_scheme', 'n_particles', 'n_nodes']).mean()[['distance', 'n_detections']])


if __name__ == '__main__':
    path = '../Experiments/DistributedBBA/results/'
    analyse_results(path)
