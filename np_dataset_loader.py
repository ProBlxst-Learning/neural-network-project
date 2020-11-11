
import numpy as np
import keras
import pandas as pd

from itertools import chain


def load_numpy():

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    np_arr = []
    l = []

    for i in range(100):
        _1d = list(chain.from_iterable(x_train[i]))
        _1d.insert(0, y_train[i])
        l.append(_1d)

    df = pd.DataFrame(l, columns=['label', 'pixel' + str(i) for i in range(784)])

    print(df)

    return x_train, y_train, x_test, y_test


def main():

    x_train, y_train, x_test, y_test = load_numpy()

    np_x_train = np.array(x_train)


if __name__ == '__main__':
    main()
