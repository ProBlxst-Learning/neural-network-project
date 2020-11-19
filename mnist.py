"""
Implementation of a dense artifical neural network with keras (tensorflow)
Inspired by:
https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
"""

################### Imports ###################

import keras
import numpy as np
import pandas as pd

import dense
import matplotlib

################### Global variables ###################

VERBOSE = True

################### Functions ###################

# this function loads the dataset from tensorflow


def load_dataset():

    (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()

    if isinstance(train_x, list):
        print("List")
    elif isinstance(train_x, np.ndarray):
        print("Numpy")
        print(train_x.shape)
    else:
        raise Exception('wrong type')

    return train_x, train_y, test_x, test_y

# the input data needs to be a float32 between 0 and 1 and the output data needs to be a categorical


def format_data(data_x, data_y):

    # convert input data to float32 and normalise
    data_x_float = data_x.astype('float32')
    input_norm = data_x_float/255.0

    # flatten pictures to one dimention
    dim = len(data_x)
    input_flatt = input_norm.reshape((dim, 28*28))

    # output data is categorised
    output_cat = keras.utils.to_categorical(data_y)

    print(input_flatt[0])
    print(input_flatt[0].shape)

    return input_flatt, output_cat

# main function for using the nn to train on mnist


def main():

    # import and format data
    train_x, train_y, test_x, test_y = load_dataset()
    train_x, train_y = format_data(train_x, train_y)
    test_x, test_y = format_data(test_x, test_y)

    # create model
    model1 = dense.NN(28*28, 10, 13)
    model2 = dense.NN(28*28, 10, 16)
    model3 = dense.NN(28*28, 10, 22)
    models = [[model1], [model2], [model3]]

    # list to collect training results
    training_results = list()

    # variable initaited to keep track of what model is currently under used
    n = 1

    # every model is trained and the data collected in the same way
    for model in models:

        # bit capacity of model
        capacity = model[0].bit_capacity()
        print('model', n, ' has a bit capacity of: ', capacity)
        model.append(capacity)

        # fit model
        results = model[0].train(train_x, train_y, test_x, test_y, epochs=1)
        # appends the test accuracy and the number of the model to the training results
        result_model = [results[2], 'bit capacity: ' + str(model[1])]
        training_results.append(result_model)
        n += 1

    # visualise training
    model1.compare_training(measures=training_results,
                            title='Training accuracy per epoch of the MNIST data set', type_measure='test accuracy')


# function used to understand the dataset (not to be used for other than development and testing)
def test():
    train_x, train_y, test_x, test_y = load_dataset()
    print('X:%s, Y:%s' % (train_x[0], train_y[0]))
    print('X:%s, Y:%s' % (1, keras.utils.to_categorical(train_y)[0]))

    x, y = format_data(train_x, train_y)
    print('\nX:%s, Y:%s' % (x[0], y[0]))

################### Main ###################


train_x, train_y, test_x, test_y = load_dataset()
format_data(train_x, train_y)
