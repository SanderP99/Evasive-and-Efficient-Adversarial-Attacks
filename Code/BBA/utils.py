import numpy as np


def line_search_to_boundary(bb_model, x_orig, x_start, label, is_targeted):
    eps = 0.4
    i = 0
    x1 = np.float32(x_start)
    x2 = np.float32(x_orig)
    diff = x2 - x1
    x_candidate = None
    while np.linalg.norm(diff) > eps:
        i += 1
        x_candidate = x1 + 0.5 * diff
        if (np.argmax(bb_model.predict(np.expand_dims(x_candidate, axis=0))) == label) == is_targeted:
            x1 = x_candidate
        else:
            x2 = x_candidate

        diff = x2 - x1
    print("Found decision boundary after {} queries.".format(i))
    print(f"Class of image is {label}, target is {label - 1}")
    print(f"Distance to original is {np.linalg.norm(x_orig - x1)}")
    return x1


def find_closest_img(bb_model, X_orig, X_targets, label, is_targeted):
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

    X_orig_normed = np.float32(X_orig)
    dists = np.empty(len(X_targets), dtype=np.float32)
    for i in range(len(X_targets)):
        d_l2 = np.linalg.norm((np.float32(X_targets[i, ...]) - X_orig_normed))
        dists[i] = d_l2

    indices = np.argsort(dists)
    for index in indices:
        X_target = X_targets[index]
        pred_clsid = np.argmax(bb_model.predict(X_target.reshape((1, 28, 28, 1))))
        if (pred_clsid == label) == is_targeted:
            print("Found an image of the target class, d_l2={:.3f}.".format(dists[index]))
            return X_target

        print("Image of target class is wrongly classified by model, skipping.")

    raise ValueError("Could not find an image of the target class that was correctly classified by the model!")
