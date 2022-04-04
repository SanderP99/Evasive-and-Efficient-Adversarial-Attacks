import matplotlib.pyplot as plt
import numpy as np

from Attacks.TargetedBBA.sampling_provider import create_perlin_noise


def plot_gaussian_vs_perlin() -> None:
    fig, gaussian_ax = plt.subplots(1, 1, figsize=(4, 4))
    fig2, perlin_ax = plt.subplots(1, 1, figsize=(4, 4))

    fig_shape = np.array([100, 100, 1])

    gaussian_ax.axis('off')
    gaussian = np.random.uniform(0, 1, fig_shape)
    gaussian_ax.imshow(gaussian, cmap='Greys')

    perlin_ax.axis('off')
    perlin = create_perlin_noise(fig_shape, freq=20)
    perlin_ax.imshow(perlin, cmap='Greys')

    fig.tight_layout()
    fig2.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_gaussian_vs_perlin()
