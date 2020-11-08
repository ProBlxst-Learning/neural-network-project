"""
Implementation of a dense artifical neural network with keras (tensorflow)
"""

################### Imports ###################

from keras.datasets import mnist

################### Global variables ###################

VERBOSE = True

(train_x, train_y), (test_x, test_y) = mnist.load_data()

################### Functions ###################


################### Main ###################

if __name__ == "__main__":
    print('X:%s, Y:%s' % (train_x[0], train_y[0]))