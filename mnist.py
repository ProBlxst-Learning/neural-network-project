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
def format_data(input, output):

    # convert input data to float32
    input_norm = input.astype('float32')

    # normalize it to the range of 0 to 1
    input_norm = input_norm/255.0

    # output data is categorised
    output_cat = keras.utils.to_categorical(output)

    return input_norm, output_cat



################### Main ###################

if __name__ == "__main__":

    # import dataset
    train_x, train_y, test_x, test_y = load_dataset()
    
    # create model
