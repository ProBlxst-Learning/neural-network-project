"""
Implementation of a dense artifical neural network with keras (tensorflow)
Influenced by:
https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
"""

################### Imports ###################

import keras
import sklearn
import matplotlib

################### Global variables ###################

VERBOSE = True

################### Objects ###################

# input_size decides the dimention of the input layer
# output_size decides the dimention of the output layer
# every argument following denotes the size of the hidden layers starting right after the input layer
class NN():

    # initial class set up
    def __init__(self, input_size, output_size, *hidden_layers):

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

        # history of training and accuracy
        self.__histories = list()
        self.__scores = list()

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
    def evaluate(self, data_x, data_y, n_folds=5):
        
        # we wish to se the progress the model went through in the training, both in traing performed and the accuracy it achieved
        history = []
        score = []
        fit = 1

        # the traing set is splitt into folds; 
        # groupings of data to be used as validation data once while the other folds acts as training data
        kfold = sklearn.model_selection.KFold(n_splits=n_folds, shuffle=True, random_state=1)

        # all combinations of the folds should be used for training and validation
        for train_ix, test_ix in kfold.split(data_x):

            # create actual data for training, based on rows chosen by kfold
            train_x, train_y, test_x, test_y = data_x[train_ix], data_y[train_ix], data_x[test_ix], data_y[test_ix]

            # train the model and record the proces
            history.append(self.__nn.fit(train_x, train_y, epochs=10, batch_size=32, validation_data=(test_x, test_y), verbose=0))

            # for every fit call, print the accuracy and recort it
            acc = self.__nn.evaluate(test_x, test_y)
            print('>%s: %.3f' % (fit, acc))
            score.append(acc)
            fit += 1

        # the training needs to be added to the history of the model
        self.__histories.append(history)
        self.__scores.append(score)

        # the model is now trained and the diagnistics should be returned
        return history, score

    
    # should visualise the training
    def visualise_training(self, history=self.__histories, scores=self.__scores):

        for i in range(len(history)):
            # plot loss
            matplotlib.pyplot.subplot(2, 1, 1)
            matplotlib.pyplot.title('Cross Entropy Loss')
            matplotlib.pyplot.plot(history[i].history['loss'], color='blue', label='train')
            matplotlib.pyplot.plot(history[i].history['val_loss'], color='orange', label='test')
            # plot accuracy
            matplotlib.pyplot.subplot(2, 1, 2)
            matplotlib.pyplot.title('Classification Accuracy')
            matplotlib.pyplot.plot(history[i].history['accuracy'], color='blue', label='train')
            matplotlib.pyplot.plot(history[i].history['val_accuracy'], color='orange', label='test')
        matplotlib.pyplot.show()

        # print summary
        #print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
        # box and whisker plots of results
        matplotlib.pyplot.boxplot(scores)
        matplotlib.pyplot.show()
        

################### Main ###################

if __name__ == "__main__":
    pass