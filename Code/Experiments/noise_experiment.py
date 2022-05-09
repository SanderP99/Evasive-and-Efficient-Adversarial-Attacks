import numpy as np
from tqdm import tqdm

from Attacks.DistributedBBA.node import Node
from Attacks.TargetedBBA.sampling_provider import create_perlin_noise
from MNIST.setup_cifar import CIFAR
from MNIST.setup_mnist import MNIST


def noise_experiment(dataset, noise_type='uniform', n_calls=10000):
    np.random.seed(42)
    node = Node(0, dataset, weights_path_mnist=f'../Defense/{dataset.upper()}Attackencoder.h5', output=True,
                flush_buffer_after_detection=False)
    if dataset == 'mnist':
        shape = (28, 28, 1)
        data = MNIST()
    elif dataset == 'cifar':
        shape = (32, 32, 3)
        data = CIFAR()
    else:
        raise ValueError

    dists = np.zeros(n_calls - 50)
    for i in tqdm(range(n_calls)):
        if noise_type == 'uniform':
            test = np.random.uniform(0, 1, shape)
        elif noise_type == 'perlin_fixed':
            test = create_perlin_noise(np.array(shape))
        elif noise_type == 'perlin':
            try:
                test = create_perlin_noise(np.array(shape), freq=np.random.randint(1, 40))
            except RuntimeWarning:
                test = create_perlin_noise(np.array(shape), freq=np.random.randint(1, 40))
        elif noise_type == 'mixed':
            idx = np.random.randint(0, 2)
            if idx == 0:
                test = data.train_data[np.random.randint(0, 2000)]
            elif idx == 1:
                test = np.random.uniform(0, 1, shape)
            elif idx == 2:
                test = create_perlin_noise(np.array(shape), freq=np.random.randint(1, 40))
        elif noise_type == 'train_data':
            test = data.train_data[np.random.randint(0, 2000)]
        else:
            raise ValueError

        result = node.add_to_detector(test)
        if len(result):
            dists[i - 50] = result[0]
    print(dataset, noise_type, dists.mean(), len(node.detector.get_detections()))


if __name__ == '__main__':
    for dataset in ['mnist', 'cifar']:
        for noise_type in ['uniform', 'mixed', 'perlin', 'perlin_fixed', 'train_data']:
            noise_experiment(dataset, n_calls=20000, noise_type=noise_type)
