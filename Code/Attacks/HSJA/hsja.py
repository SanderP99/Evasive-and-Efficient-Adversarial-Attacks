import csv
from typing import Optional, Union

import numpy as np
from keras.models import Model

from Attacks.DistributedBBA.node import Node


class HopSkipJumpAttack:

    def __init__(self, model: Model):
        self.blackbox_model: Model = model
        self.calls: int = 0
        self.dimensions = None
        self.d: int = 0
        self.sqrt_d: float = 0
        self.theta: float = 0
        self.is_targeted: bool = False
        self.label: int = -1

    def run_attack(self, x_orig: np.ndarray, label: int, is_targeted: bool, x_start: np.ndarray, n_calls_left,
                   n_max_per_batch: int = 50, n_seconds: Optional[int] = None, source_step: float = 1e-2,
                   spherical_step: float = 1e-2, mask: Optional[np.ndarray] = None,
                   recalc_mask_every=None, pso: bool = False, output: bool = False, filename: Optional[str] = None,
                   node: Optional[Node] = None, maximal_calls: int = 10000, dimensions: np.ndarray = None,
                   num_iterations: int = 40
                   ) -> Optional[np.ndarray]:
        self.calls = 0
        self.dimensions = dimensions
        self.d = int(np.prod(self.dimensions))
        self.sqrt_d = np.sqrt(self.d)
        self.theta = 1 / (self.d * self.sqrt_d)
        self.is_targeted = is_targeted
        self.label = label

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

        # Initialize
        perturbed = self.initialize(x_orig, x_start)
        distance = self.compute_distance(x_orig, perturbed)

        for i in np.arange(num_iterations):
            delta = self.select_delta(i, distance)
            num_evals = int(np.sqrt(i + 1) * 100)
            num_evals = np.min(1e4, num_evals)
            gradf = self.approximate_gradient(perturbed, num_evals, delta)
            
            epsilon = self.geometric_progression_for_stepsize(perturbed, gradf, distance)

    def initialize(self, x_orig: np.ndarray, x_start: np.ndarray) -> np.ndarray:
        n_evals: int = 0

        if x_start is None:
            while True:
                random_noise = np.random.uniform(0, 1, size=self.dimensions)
                success = self._eval_sample(random_noise)
                n_evals += 1
                if success:
                    break
                assert n_evals < 1e4, "Init failed!"

            low = 0
            high = 1
            while high - low > 0.001:
                mid = (high - low) / 2
                blended = (1 - mid) * x_orig + mid * random_noise
                success = self._eval_sample(blended)
                if success:
                    high = mid
                else:
                    low = mid
            init = (1 - high) * x_orig + high * random_noise
        else:
            init = x_start

        self.calls += n_evals
        return init

    def _eval_sample(self, sample: np.ndarray) -> bool:
        pred = self.blackbox_model.predict(sample.reshape((1,) + self.dimensions))
        pred_label = np.argmax(pred)
        if (pred_label == self.label) != self.is_targeted:
            # No longer adversarial
            return False
        else:
            return True
        
    def _eval_samples(self, samples: np.ndarray) -> np.ndarray:
        result = []
        for sample in samples:
            result += self._eval_sample(sample)
        return np.array(result)

    def compute_distance(self, x_orig: np.ndarray, perturbed: np.ndarray) -> float:
        return np.linalg.norm(x_orig - perturbed)

    def select_delta(self, iteration: int, distance: float) -> float:
        if iteration == 0:
            delta = 0.1 * (1 - 0)
        else:
            delta = self.sqrt_d * self.theta * distance
        return delta

    def approximate_gradient(self, sample: np.ndarray, num_evals: int, delta: float):
        noise_shape = [num_evals] + list(self.dimensions)
        rv = np.random.randn(*noise_shape)
        rv /= np.sqrt(np.sum(rv ** 2, axis=(1, 2, 3), keepdims=True))
        perturbed = sample + delta * rv
        perturbed = np.clip(perturbed, 0, 1)
        rv = (perturbed - sample) / delta

        # Query the model
        decisions = self._eval_samples(perturbed)
        decisions_shape = [len(decisions)] + [1] * len(self.dimensions)
        fval = 2 * decisions.astype(float).reshape(decisions_shape) - 1.0
        
        # Baseline subtraction
        if np.mean(fval) == 1.0:
            gradf = np.mean(rv, axis=0)
        elif np.mean(fval) == -1.0:
            gradf = -np.mean(rv, axis=0)
        else:
            fval -= np.mean(fval)
            gradf = np.mean(fval * rv, axis=0)
        
        gradf /= np.linalg.norm(gradf)
        return gradf

    def geometric_progression_for_stepsize(self, perturbed, gradf, distance):
        pass
        
