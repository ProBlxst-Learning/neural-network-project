"""
Implementation of a dense artifical neural network with keras (tensorflow)
Inspired by:
https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
"""

################### Imports ###################

import keras
import numpy
import dense

################### Global variables ###################

VERBOSE = True

################### Functions ###################

# this function loads the dataset from tensorflow


def load_dataset():

    (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()

    return train_x, train_y, test_x, test_y

# the input data needs to be a float32 between 0 and 1 and the output data needs to be a categorical


def format_data(data_x, data_y):

    # convert input data to float32 and normalise
    data_x_float = data_x.astype('float32')
    input_norm = data_x_float/255.0

    # flatten pictures to one dimention
    dim = len(data_x)
    input_flatt = input_norm.reshape((dim,28*28))

    # output data is categorised
    output_cat = keras.utils.to_categorical(data_y)

    return input_flatt, output_cat

# main function for using the nn to train on mnist


def main():

    # import and format data
    train_x, train_y, test_x, test_y = load_dataset()
    train_x, train_y = format_data(train_x, train_y)
    test_x, test_y = format_data(test_x,test_y)

    # create model
    model = dense.NN(28*28, 10)

    # bit capacity of model
    print(model.bit_capacity())

    # fit model
    model.evaluate(train_x, train_y, test_x, test_y)

    # visualise training
    model.visualise_training()


# function used to understand the dataset (not to be used for other than development and testing)
def test():
    train_x, train_y, test_x, test_y = load_dataset()
    print('X:%s, Y:%s' % (train_x[0], train_y[0]))
    print('X:%s, Y:%s' % (1, keras.utils.to_categorical(train_y)[0]))

    x, y = format_data(train_x, train_y)
    print('\nX:%s, Y:%s' % (x[0], y[0]))

################### Main ###################


if __name__ == "__main__":
    main()
