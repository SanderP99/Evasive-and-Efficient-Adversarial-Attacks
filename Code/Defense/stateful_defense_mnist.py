import numpy as np

from Defense.detector import SimilarityDetector
from MNIST.setup_mnist import MNIST


def create_stateful_mnist_defence():
    # mnist = MNIST()
    # x_train = np.array(mnist.train_data)
    # 0.449137 = threshold
    detector = SimilarityDetector(50, threshold=0.449137, weights_path='encoder_weights.h5')
    return detector


if __name__ == '__main__':
    detector = create_stateful_mnist_defence()
    mnist = MNIST()
    x_train = np.array(mnist.train_data)
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
