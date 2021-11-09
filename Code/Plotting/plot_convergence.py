import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from MNIST.setup_mnist import MNIST

sns.set()
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def main():
    df_pso = pd.read_csv('../BBA/distance_files/distances_pso_1302_3_5.csv')
    df_bba = pd.read_csv('../BBA/distance_files/distances_bba_1302_3.csv')
    df_pso_10 = pd.read_csv('../BBA/distance_files/distances_pso_1302_3_10.csv')

    fig, ax = plt.subplots(1)

    ax.plot(df_bba['Queries'], df_bba['Distance'], label='BBA')
    ax.plot(df_pso['Queries'], df_pso['Distance'], label='PSO (5 particles)')
    ax.plot(df_pso_10['Queries'], df_pso_10['Distance'], label='PSO (10 particles)')
    ax.legend(loc='upper center')
    ax.set_xlabel('# Queries')
    ax.set_ylabel('L2 distance to original')

    inset_ax = inset_axes(ax, width=1, height=1)
    image = MNIST().test_data[1302]
    inset_ax.imshow(image, cmap='gray')
    inset_ax.axis('off')

    ax.set_title('Distance to original image while predicting 3')
    plt.show()


if __name__ == '__main__':
    main()
