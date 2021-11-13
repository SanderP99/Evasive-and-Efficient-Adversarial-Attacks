import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from MNIST.setup_mnist import MNIST

sns.set()
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def main():
    i = 6265
    t = 2
    df_pso = pd.read_csv(f'../BBA/distance_files/distances_pso_{i}_{t}_5_random.csv')
    df_bba = pd.read_csv(f'../BBA/distance_files/distances_pso_{i}_{t}_5_closest.csv')
    df_pso_10 = pd.read_csv(f'../BBA/distance_files/distances_pso_{i}_{t}_10_random.csv')
    df_pso_10c = pd.read_csv(f'../BBA/distance_files/distances_pso_{i}_{t}_10_closest.csv')

    fig, ax = plt.subplots(1)

    ax.plot(df_bba['Queries'], df_bba['Distance'], label='PSO (5 particles, closest)')
    ax.plot(df_pso['Queries'], df_pso['Distance'], label='PSO (5 particles, random)')
    ax.plot(df_pso_10c['Queries'], df_pso_10c['Distance'], label='PSO (10 particles, closest)')
    ax.plot(df_pso_10['Queries'], df_pso_10['Distance'], label='PSO (10 particles, random)')
    ax.legend(loc='upper center')
    ax.set_xlabel('# Queries')
    ax.set_ylabel('L2 distance to original')

    inset_ax = inset_axes(ax, width=1, height=1)
    image = MNIST().test_data[i]
    inset_ax.imshow(image, cmap='gray')
    inset_ax.axis('off')

    ax.set_title('Distance to original image while predicting 7')
    plt.show()


if __name__ == '__main__':
    main()
