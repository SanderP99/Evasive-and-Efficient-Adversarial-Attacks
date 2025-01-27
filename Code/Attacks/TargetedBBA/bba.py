from typing import Optional, Union, Tuple

import numpy as np
import csv

from keras.models import Model

from Attacks.DistributedBBA.node import Node
from Attacks.TargetedBBA.sampling_provider import create_perlin_noise


class BiasedBoundaryAttack:

    def __init__(self, model: Model, sample_gen):
        self.blackbox_model: Model = model
        self.sample_gen = sample_gen
        self.calls: int = 0
        self.dimensions = None

    def run_attack(self, x_orig: np.ndarray, label: int, is_targeted: bool, x_start: np.ndarray, n_calls_left,
                   n_max_per_batch: int = 50, n_seconds: Optional[int] = None, source_step: float = 1e-2,
                   spherical_step: float = 1e-2, mask: Optional[np.ndarray] = None,
                   recalc_mask_every=None, pso: bool = False, output: bool = False, filename: Optional[str] = None,
                   node: Optional[Node] = None, maximal_calls: int = 10000, dimensions: np.ndarray = None) -> Optional[
        np.ndarray]:
        self.calls = 0
        self.dimensions = dimensions
        if output:
            assert filename is not None
            self.calls = maximal_calls - n_calls_left()
            with open(filename, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(['Queries', 'Distance'])
            with open(filename, 'a') as file:
                writer = csv.writer(file)
                writer.writerow([self.calls, np.linalg.norm(x_orig - x_start)])
        assert len(x_orig.shape) == 3
        assert len(x_start.shape) == 3

        if mask is not None:
            assert mask.shape == x_orig.shape
            assert np.sum(mask < 0) == 0 and 1. - np.max(
                mask) < 1e-4, "Mask must be scaled to [0,1]. At least one value must be 1."
        else:
            mask = np.ones(x_orig.shape, dtype=np.float32)

        current_label, best_distance = self._eval_sample(x_start, x_orig, node=node)
        if (current_label == label) != is_targeted:
            print("WARN: Starting point is not a valid adversarial example! Continuing for now.")
            return
        x_adv_best = np.copy(x_start)
        last_mask_recalc_calls = n_calls_left()

        while n_calls_left() > 0:
            if output:
                with open(filename, 'a') as file:
                    writer = csv.writer(file)
                    writer.writerow([self.calls, np.linalg.norm(x_orig - x_adv_best)])
            if recalc_mask_every is not None and last_mask_recalc_calls - n_calls_left() >= recalc_mask_every:
                new_mask = np.abs(x_adv_best - x_orig)
                new_mask /= np.max(new_mask)  # scale to [0,1]
                new_mask = new_mask ** 0.5  # weaken the effect a bit.
                print(
                    "Recalculated mask. Weighted dimensionality of search space is now {:.0f} (diff: {:.2%}). ".format(
                        np.sum(new_mask), 1. - np.sum(new_mask) / np.sum(mask)))
                mask = new_mask
                last_mask_recalc_calls = n_calls_left()

            n_candidates = min(n_max_per_batch, n_calls_left())

            for i in range(n_candidates):
                candidate = self.generate_candidate(i, n_candidates, x_orig, x_adv_best, mask, source_step,
                                                    spherical_step)
                candidate_label, dist = self._eval_sample(candidate, x_orig, node=node)
                # print(candidate_label, label)
                if (candidate_label == label) == is_targeted:
                    # print(dist, best_distance)
                    if dist < best_distance:
                        x_adv_best = candidate
                        best_distance = dist
                        break
            if pso:
                return candidate
        return x_adv_best

    def _eval_sample(self, x: np.ndarray, x_orig_normed: Optional[np.ndarray] = None, node: Optional[Node] = None) -> \
            Union[int, Tuple[int, float]]:
        pred = self.blackbox_model(x.reshape((1,) + self.dimensions))
        if node is not None:
            node.add_to_detector(x.reshape(self.dimensions))
        self.calls += 1
        label = np.argmax(pred)

        if x_orig_normed is None:
            return label
        else:
            d = np.linalg.norm(x - x_orig_normed)
            return label, d

    def generate_candidate(self, i: int, n: int, x_orig: np.ndarray, x_adv_best: np.ndarray, mask: np.ndarray,
                           source_step: float, spherical_step: float) -> np.ndarray:
        # Scale both spherical and source step with i.
        scale = (1. - i / n) + 0.3
        c_source_step = source_step * scale
        c_spherical_step = spherical_step * scale
        sampling_fn = create_perlin_noise

        candidate = self.generate_boundary_sample(x_orig, x_adv_best, mask, c_source_step, c_spherical_step,
                                                  sampling_fn)

        return candidate

    def generate_boundary_sample(self, x_orig: np.ndarray, x_adv_best: np.ndarray, mask: np.ndarray, source_step: float,
                                 spherical_step: float, sampling_fn) -> np.ndarray:
        unnormalized_source_direction = np.float32(x_orig) - np.float32(x_adv_best)
        source_norm = np.linalg.norm(unnormalized_source_direction)
        source_direction = unnormalized_source_direction / source_norm

        # Get perturbation from provided distribution
        sampling_dir = sampling_fn(x_orig.shape)

        # ===========================================================
        # calculate candidate on sphere
        # ===========================================================
        dot = np.vdot(sampling_dir, source_direction)
        sampling_dir -= dot * source_direction  # Project orthogonal to source direction
        sampling_dir *= mask  # Apply regional mask
        sampling_dir /= np.linalg.norm(sampling_dir)  # Norming increases magnitude of masked regions

        sampling_dir *= spherical_step * source_norm  # Norm to length stepsize*(dist from src)

        D = 1 / np.sqrt(spherical_step ** 2 + 1)
        direction = sampling_dir - unnormalized_source_direction
        spherical_candidate = x_orig + D * direction

        np.clip(spherical_candidate, 0., 1., out=spherical_candidate)

        # ===========================================================
        # step towards source
        # ===========================================================
        new_source_direction = x_orig - spherical_candidate

        new_source_direction_norm = np.linalg.norm(new_source_direction)
        new_source_direction /= new_source_direction_norm
        spherical_candidate = x_orig - source_norm * new_source_direction  # Snap sph.c. onto sphere

        # From there, take a step towards the target.
        candidate = spherical_candidate + (source_step * source_norm) * new_source_direction

        np.clip(candidate, 0., 1., out=candidate)
        return np.float32(candidate)
