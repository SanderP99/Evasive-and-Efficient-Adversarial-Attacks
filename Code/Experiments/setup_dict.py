import numpy as np
import pandas as pd
from keras.models import load_model

from MNIST.setup_cifar import CIFAR
from MNIST.setup_mnist import MNIST


def make_predictions(mnist, model):
    preds = []
    for x in mnist.test_data:
        preds.append(np.argmax(model.predict(np.expand_dims(x, axis=0))))
    return preds


def create_dataframe(mnist, preds):
    data = {}
    data['real_labels'] = np.argmax(mnist.test_labels, axis=1)
    data['predictions'] = preds
    df = pd.DataFrame(data=data)
    df['correct'] = (df['real_labels'] == df['predictions']).astype(int)

    return df


def split_dataframe(df):
    dfs = []
    for l in range(10):
        dfl = df[df['correct'] == 1]
        dfl = dfl[dfl['real_labels'] == l]
        dfs.append(dfl)
    return dfs


if __name__ == '__main__':
    np.random.seed(42)
    # cifar = CIFAR()
    # model = load_model('../MNIST/models/cifar', compile=False)
    # predictions = make_predictions(cifar, model)
    # df = create_dataframe(cifar, predictions)
    # df.to_csv('predictions_cifar.csv', index_label='index')
    df = pd.read_csv('predictions_cifar.csv', index_col='index')
    dfs = split_dataframe(df)
    experiments = {}
    i = 0
    indexes = []
    targets = []
    y_orig = []
    y_target = []
    while i < 500:
        random_index = np.random.randint(0, 10000)
        if df.iloc[random_index].correct == 1:
            i += 1
            random_target = np.random.randint(0, 10)
            while random_target == df.iloc[random_index].real_labels:
                random_target = np.random.randint(0, 10)
            indexes.append(random_index)
            targets.append(dfs[random_target].index.values.tolist())
            y_orig.append(df.iloc[random_index].real_labels)
            y_target.append(random_target)
    experiments['index'] = indexes
    experiments['targets'] = targets
    experiments['y_orig'] = y_orig
    experiments['y_target'] = y_target
    experiments_df = pd.DataFrame(data=experiments)
    experiments_df.to_csv('experiments_cifar.csv', index=False)
#