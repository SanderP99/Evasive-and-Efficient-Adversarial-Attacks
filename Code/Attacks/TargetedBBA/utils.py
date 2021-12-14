from typing import Union, Optional, Tuple

import numpy as np
from keras.models import Model


def line_search_to_boundary(bb_model: Model, x_orig: np.ndarray, x_start: np.ndarray, label: int, is_targeted: bool,
                            calls: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
    eps: float = 0.4
    i: int = 0
    x1: np.ndarray = np.float32(x_start)
    x2: np.ndarray = np.float32(x_orig)
    diff: np.ndarray = x2 - x1
    x_candidate: Optional[np.ndarray] = None
    while np.linalg.norm(diff) > eps:
        i += 1
        x_candidate = x1 + 0.5 * diff
        if (np.argmax(bb_model.predict(np.expand_dims(x_candidate, axis=0))) == label) == is_targeted:
            x1 = x_candidate
        else:
            x2 = x_candidate

        diff = x2 - x1
    # print("Found decision boundary after {} queries.".format(i))
    # print(f"Class of image is {label}, target is {label - 1}")
    # print(f"Distance to original is {np.linalg.norm(x_orig - x1)}")
    if calls:
        return x1, i
    return x1


def find_closest_img(bb_model: Model, X_orig: np.ndarray, X_targets: np.ndarray, label: int,
                     is_targeted: bool) -> (np.ndarray, int):
    """
    From a list of potential starting images, finds the closest to the original.
    Before returning, this method makes sure that the image fulfills the adversarial condition (is actually classified as the target label).
    :param bb_model: The (black-box) model.
    :param X_orig: The original image to attack.
    :param X_targets: List of images that fulfill the adversarial criterion (i.e. target class in the targeted case)
    :param is_targeted: true if the attack is targeted.
    :param label: the target label if targeted, or the correct label if untargeted.
    :return: the closest image (in L2 distance) to the original that also fulfills the adversarial condition.
    """

    X_orig_normed: np.ndarray = np.float32(X_orig)
    dists: np.ndarray = np.empty(len(X_targets), dtype=np.float32)
    for i in range(len(X_targets)):
        d_l2 = np.linalg.norm((np.float32(X_targets[i, ...]) - X_orig_normed))
        dists[i] = d_l2

    indices: np.ndarray = np.argsort(dists)
    calls: int = 0
    for index in indices:
        X_target = X_targets[index]
        pred_clsid = np.argmax(bb_model.predict(X_target.reshape((1, 28, 28, 1))))
        calls += 1
        if (pred_clsid == label) == is_targeted:
            print("Found an image of the target class, d_l2={:.3f}.".format(dists[index]))
            return X_target, calls

        print("Image of target class is wrongly classified by model, skipping.")

    raise ValueError("Could not find an image of the target class that was correctly classified by the model!")
