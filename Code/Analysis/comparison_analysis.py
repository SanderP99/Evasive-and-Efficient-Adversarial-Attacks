import pandas as pd

from Analysis.result_analysis_comparison import compare


def comparison_analysis(file):
    x = pd.read_csv(file)
    x.columns = [name.strip() for name in x.columns]
    x['gain'] = x['n_detections_split'] - x['n_detections_combined']
    print(x.groupby(['dataset', 'n_experiments']).mean())


if __name__ == '__main__':
    path = '../Experiments/DistributedBBA/results/'
    compare(path)
    comparison_analysis('combination_comparison.csv')