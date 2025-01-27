from typing import Optional

import numpy as np

np.random.seed(42)


# From: https://github.com/ttbrunner/biased_boundary_attack/blob/0a810b51e6039876c8626ca19d8575b2864f531a/utils/sampling/perlin.py

def create_perlin_noise(px: np.ndarray, seed: Optional[int] = None, color: Optional[bool] = False,
                        normalize: Optional[bool] = True, freq: int = 5, precalc_fade=None):
    """
    Creates a batch of perlin noise patterns.
    :param px: image dimension. Only quadratic images are supported, i.e. (px * px * 3).
    :param seed: random seed (optional).
    :param color: if True, generates color noise. If False, generates black-and-white noise (still, the output has 3 identical channels).
    :param normalize: if True, normalizes each sample to have a L2-norm of 1.
    :param freq: noise frequency. 5 seems to work OK for Tiny ImageNet (64x64), while 20 seems to be better for ImageNet (299x299).
    :param precalc_fade: if not None, uses the provided fade factors for faster calculation.
    :return: batch of samples (b, px, px, 3)
    """

    if seed is not None:
        np.random.seed(seed)

    output = np.zeros(px, dtype=np.float32)

    if px[2] == 3:
        color = True
    while np.all(output == 0):

        if color:
            # Draw 3 separate Perlin samples per image, one for each channel
            batch = perlin_batch(px[0], n_samples=3, freq=freq, precalc_fade=precalc_fade)
            for i in range(1):
                output[:, :, 0] = batch[i * 3]
                output[:, :, 1] = batch[i * 3 + 1]
                output[:, :, 2] = batch[i * 3 + 2]
        else:
            # Draw 1 Perlin sample per image and repeat it
            batch = perlin_batch(px[0], n_samples=1, freq=freq, precalc_fade=precalc_fade)
            for i in range(1):
                output[:, :, 0] = batch[i]
        freq += 1

    # Normalize to L2=1
    if normalize:
        if np.linalg.norm(output.reshape(1, -1), axis=1) == [0.]:
            print(output)
        output /= np.linalg.norm(output.reshape(1, -1), axis=1)
    return output


def calc_fade(n_pixels_side, freq=5):
    # Precalc fade factors for faster generation

    lin = np.linspace(0, freq, n_pixels_side, endpoint=False)
    x, y = np.meshgrid(lin, lin)

    # coordinates of the top-left
    xi = x.astype(int)
    yi = y.astype(int)
    # internal coordinates
    xf = x - xi
    yf = y - yi
    # fade factors
    u = fade(xf)
    v = fade(yf)
    return xi, yi, xf, yf, u, v


def perlin_batch(n_pixels_side, n_samples, freq=5, precalc_fade=None):
    # Deliver a batch of Perlin noise. This is partially vectorized, so it should be slightly faster.

    if precalc_fade is None:
        precalc_fade = calc_fade(n_pixels_side, freq=freq)
    xi, yi, xf, yf, u, v = precalc_fade

    # Permutation table, which is randomly shuffled on each call.
    p = np.repeat(np.arange(256, dtype=int)[np.newaxis, :], n_samples, axis=0)
    for i in range(n_samples):
        np.random.shuffle(p[i])
    p = np.tile(p, (1, 2))

    n00 = np.empty((n_samples, n_pixels_side, n_pixels_side), dtype=np.float32)
    n01 = np.empty((n_samples, n_pixels_side, n_pixels_side), dtype=np.float32)
    n11 = np.empty((n_samples, n_pixels_side, n_pixels_side), dtype=np.float32)
    n10 = np.empty((n_samples, n_pixels_side, n_pixels_side), dtype=np.float32)
    for i in range(n_samples):
        n00[i] = gradient(p[i, p[i, xi] + yi], xf, yf)
        n01[i] = gradient(p[i, p[i, xi] + yi + 1], xf, yf - 1)
        n11[i] = gradient(p[i, p[i, xi + 1] + yi + 1], xf - 1, yf - 1)
        n10[i] = gradient(p[i, p[i, xi + 1] + yi], xf - 1, yf)

    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)
    return lerp(x1, x2, v)


def lerp(a, b, x):
    return a + x * (b - a)


def fade(t):
    # 6t^5 - 15t^4 + 10t^3
    return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3


def gradient(h, x, y):
    # grad converts h to the right gradient vector and return the dot product with (x,y)
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    g = vectors[h % 4]
    return g[:, :, 0] * x + g[:, :, 1] * y
