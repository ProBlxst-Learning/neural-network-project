"""
Implementation of a dense artifical neural network with keras (tensorflow)
Inspired by:
https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
"""

################### Imports ###################

import keras
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
    input_flatt = input_norm.reshape((dim, 28*28))

    # output data is categorised
    output_cat = keras.utils.to_categorical(data_y)

    return input_flatt, output_cat

# main function for using the nn to train on mnist


def main():

    # import and format data
    train_x, train_y, test_x, test_y = load_dataset()
    train_x, train_y = format_data(train_x, train_y)
    test_x, test_y = format_data(test_x, test_y)

    # create model
    model1 = dense.NN(28*28, 10)
    model2 = dense.NN(28*28, 10, 16)
    model3 = dense.NN(28*28, 10, 32)
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
        results = model[0].train(train_x, train_y, test_x, test_y, epochs=5)
        # appends the test accuracy and the number of the model to the training results
        result_model = [results[2], 'bit capacity: ' + str(model[1])]
        training_results.append(result_model)
        n += 1

    # visualise training
    model1.compare_training(measures=training_results,
                            title='Training accuracy per epoch of the MNIST data set', type_measure='test accuracy')


################### Main ###################

if __name__ == "__main__":
    main()
