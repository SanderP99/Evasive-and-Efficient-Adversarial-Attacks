import numpy as np

from Defense.detector import SimilarityDetector
from MNIST.setup_cifar import CIFAR


def create_stateful_cifar_defence() -> SimilarityDetector:
    cifar = CIFAR()
    x_train = np.array(cifar.train_data)
    # 0.449137 = threshold VAE
    # 0.00926118174381554 = threshold SE MNIST
    return SimilarityDetector(50, weights_path='CIFARencoder.h5', training_data=x_train)


if __name__ == '__main__':
    detector = create_stateful_cifar_defence()
    print(detector.threshold)
    cifar = CIFAR()
    x_train = np.array(cifar.train_data)

    perm = np.random.permutation(x_train.shape[0])

    benign_queries = x_train[perm[:1000], :, :, :]
    suspicious_queries = x_train[perm[-1], :, :, :] * np.random.normal(0, 0.05, (1000,) + x_train.shape[1:])

    detector.process(benign_queries)

    detections = detector.get_detections()
    print("Num detections:", len(detections))
    print("Queries per detection:", detections)
    print("i-th query that caused detection:", detector.history)

    detector.clear_memory()
    detector.process(suspicious_queries)
    detections = detector.get_detections()
    print("Num detections:", len(detections))
    print("Queries per detection:", detections)
    print("i-th query that caused detection:", detector.history)

    detector.clear_memory()
    detector.process(x_train)
    detections = detector.get_detections()
    print("Num detections:", len(detections))
    print("Queries per detection:", detections)
    print("i-th query that caused detection:", detector.history)
