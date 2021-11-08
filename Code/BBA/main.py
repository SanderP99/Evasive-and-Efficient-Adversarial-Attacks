import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from tqdm import tqdm

from BBA.bba import BiasedBoundaryAttack
from BBA.bba_pso import ParticleBiasedBoundaryAttack
from BBA.sampling_provider import create_perlin_noise
from MNIST.setup_mnist import MNIST


def main():
    n_classes = 10
    img_shape = (28, 28, 1)
    bb_model = load_model('../MNIST/models/mnist', compile=False)
    mnist = MNIST()

    # bba_attack = BiasedBoundaryAttack(bb_model, create_perlin_noise)
    # n_calls_max = 10_000
    #
    # # Starting point
    # for i in range(42, 43):
    #     print("-" * 50)
    #     img_orig = mnist.test_data[i].reshape((1, 28, 28, 1))
    #     clsid_gt = np.argmax(mnist.test_labels[i])
    #     clsid_target = clsid_gt + 1
    #     print(clsid_gt, clsid_target)
    #     target_ids = np.arange(len(mnist.test_labels))[clsid_target == np.argmax(mnist.test_labels, axis=1)]
    #     X_targets = mnist.test_data[target_ids]
    #     x_start = find_closest_img(bb_model, img_orig, X_targets, clsid_target, True) + .5
    #     x_start = line_search_to_boundary(bb_model, img_orig, x_start, label=clsid_target, is_targeted=True)
    #     x_adv = bba_attack.run_attack(img_orig, clsid_target, True, x_start, (lambda: n_calls_max - bba_attack.calls),
    #                                   source_step=2e-2,
    #                                   spherical_step=5e-2, mask=None, recalc_mask_every=500)
    #     pred = np.argmax(bb_model.predict(x_adv.reshape((1, 28, 28, 1))))
    #     dist = np.linalg.norm(x_adv - img_orig)
    #     fig, ax = plt.subplots(1, 1)
    #     ax.imshow(x_adv)
    #     ax.set_title(f"Prediction: {pred}, distance: {dist}")
    #     plt.show()
    #     print(f"Prediction: {pred}, distance: {dist}")

    labels = np.argmax(mnist.test_labels, axis=1)
    labels = mnist.test_data[labels == 9]
    n_particles = 5
    inits = labels[:n_particles]
    swarm = ParticleBiasedBoundaryAttack(n_particles=n_particles, model=bb_model, target_img=mnist.test_data[42],
                                         target_label=9, inits=inits)
    current_best = np.infty
    for i in tqdm(range(100)):
        swarm.optimize()
        p, f = swarm.get_best_particle()

        if f < current_best:
            fig, ax = plt.subplots(1, 3)
            p, f = swarm.get_best_particle()
            current_best = f
            q = swarm.get_worst_article()[0]
            ax[0].imshow(p, cmap='gray')
            ax[1].imshow(mnist.test_data[42], cmap='gray')
            ax[2].imshow(q, cmap='gray')
            ax[0].set_title('Adversarial')
            ax[1].set_title('Original')
            ax[2].set_title('Worst')
            pred = np.argmax(bb_model.predict(np.expand_dims(p, axis=0)))
            predw = np.argmax(bb_model.predict(np.expand_dims(q, axis=0)))
            fig.suptitle(
                f'Iteration {swarm.iteration}, L2-distance: {np.round(f, 4)}, Prediction: {pred, predw}, \n Total '
                f'queries: {swarm.total_queries}')
            fig.tight_layout()
            # TODO: plot prediction for worst and compare with vanilla BA

            plt.show()


if __name__ == '__main__':
    main()
