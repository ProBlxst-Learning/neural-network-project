"""
Implementation of a dense artifical neural network with keras (tensorflow)
"""

################### Imports ###################

import keras

################### Global variables ###################

VERBOSE = True
INPUT_SIZE = 1000
OUTPUT_SIZE = 10

################### Objects ###################

# input_size decides the dimention of the input layer
# output_size decides the dimention of the output layer
# every argument following denotes the size of the hidden layers starting right after the input layer
class NN():

    # initial class set up
    def __init__(self, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, *hidden_layers):

        # save arguments in the object
        self.__layers = [input_size, hidden_layers, output_size]

        # define neural network
        self.__nn = keras.models.Sequential([

            # the input layer should be as large as the defined input size
            keras.layers.core.Dense(int(input_size), input_shape=(1,), activation='relu'),

            # make one layer per hidden layer with the specified number of neurons
            [keras.layers.core.Dense(neurons) for neurons in hidden_layers],

            # the output layer shoud be as large as the defined output size
            keras.layers.core.Dense(output_size)
        ])

        self.__nn.compile()

    ################### Methods ###################

    # should return the bit capacity of the nn
    # calculate bit capacity according to Friedland (2018)
    # assuming every dense layer contains biases and weights between every neuron in the next layer
    def bit_capacity(self):
        return -1

        # set initail capasity to 0
        self.__capacity = -1

        # initialize the bit capacity for all layers to zero
        capacities = []

        # the first layer gives a capacity of all weights pluss all biases
        #for layer in self.__layers:
            





    # used to train network with given training data
    def train(self, training_data):
        return -1
    

################### Main ###################

if __name__ == "__main__":
    pass