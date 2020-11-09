"""
Implementation of a dense artifical neural network with keras (tensorflow)
Influenced by:
https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
"""

################### Imports ###################

import keras
from matplotlib import pyplot
from sklearn.model_selection import KFold

################### Global variables ###################

VERBOSE = True

################### Objects ###################

# input_size decides the dimention of the input layer
# output_size decides the dimention of the output layer
# every argument following denotes the size of the hidden layers starting right after the input layer
class NN():

    # initial class set up
    def __init__(self, input_size, output_size, *hidden_layers):

        # save arguments in the object in the order; input, hidden, output
        self.__layers = [input_size]
        for neurons in hidden_layers:
            self.__layers.append(neurons)
        self.__layers.append(output_size)
        print(self.__layers) if VERBOSE else None

        # define neural network
        self.__nn = keras.models.Sequential()

        # the input layer should be as large as the defined input size
        self.__nn.add(keras.layers.core.Dense(int(input_size), input_dim=int(input_size), activation='relu'))

            # make one layer per hidden layer with the specified number of neurons
        [self.__nn.add(keras.layers.core.Dense(neurons, activation='relu')) for neurons in hidden_layers],

        # the output layer shoud be as large as the defined output size
        self.__nn.add(keras.layers.core.Dense(output_size, activation='softmax'))

        # compile model
        self.__nn.compile(optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9) , loss='categorical_crossentropy', metrics=['accuracy'])

        # history of training and accuracy
        self.__histories = list()
        self.__scores = list()

        print('Model made') if VERBOSE else None


    ################### Methods ###################

    """
    calculate bit capacity according to Friedland (2018)
    assuming every dense layer contains biases and weights between every neuron in the next layer

    rule 1: The output of a single perceptron yields maximally one bit of information.
    rule 2: The capacity of a single perceptron is the number of its parameters (weights and bias) in bits.
    rule 3: For perceptrons in series (e.g., in subsequent layers), the capacity of a subsequent layer cannot be larger 
            than the output of the previous layer.

    """
    # should return the bit capacity of the nn
    def bit_capacity(self):

        # seek to find the capacity for all layers, staring at the first hidden layer, and summing
        capacities = list()

        # the first layer gives a capacity of all weights pluss all biases
        # only rule 1 applies
        capacity = self.__layers[0]*self.__layers[1] # weights
        capacity += self.__layers[1] # biases
        capacities.append(capacity)
        print('bit capacity layer %s: %s' % (1, capacity)) if VERBOSE else None

        # for all but the first layer the capacity contribution is min(rule 2, output of previous layer) this is rule 3
        for i in range (2, len(self.__layers)):
            print(i)
            capacity = self.__layers[i-1]*self.__layers[i] # weights
            capacity += self.__layers[i] # biases
            capacities.append(min(capacity, self.__layers[i-1]))
            print('bit capacity layer %s: %s' % (i, capacities[i-1])) if VERBOSE else None
        
        return sum(capacities)


    # used to train network with given training data
    def evaluate(self, data_x, data_y, n_folds=5):
        
        # we wish to se the progress the model went through in the training, both in traing performed and the accuracy it achieved
        history = []
        score = []
        fit = 1

        # the traing set is splitt into folds; 
        # groupings of data to be used as validation data once while the other folds acts as training data
        print('kfold initiated') if VERBOSE else None
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=1)
        print('kfold finished') if VERBOSE else None

        print('Training initiated') if VERBOSE else None

        # all combinations of the folds should be used for training and validation
        for train_ix, test_ix in kfold.split(data_x):

            # create actual data for training, based on rows chosen by kfold
            train_x, train_y, test_x, test_y = data_x[train_ix], data_y[train_ix], data_x[test_ix], data_y[test_ix]

            # train the model and record the proces
            print('fit initiated') if VERBOSE else None
            instance = self.__nn.fit(train_x, train_y, epochs=10, batch_size=32, validation_data=(test_x, test_y), verbose=VERBOSE)
            print('fit finished') if VERBOSE else None
            history.append(instance)

            # for every fit call, print the accuracy and recort it
            print('evaluate initiated') if VERBOSE else None
            loss, acc = self.__nn.evaluate(test_x, test_y, verbose=1)
            print('evaluate finished') if VERBOSE else None
            print(acc)
            print('>%s: %.3f' % (fit, acc))
            score.append(acc)
            fit += 1

        # the training needs to be added to the history of the model
        self.__histories.append(history)
        self.__scores.append(score)

        # the model is now trained and the diagnistics should be returned
        return history, score

    
    # should visualise the training
    def visualise_training(self, history=None, scores=None):

        history = history if history else self.__histories
        scores = scores if scores else self.__scores

        for i in range(len(history)):
            # plot loss
            pyplot.subplot(2, 1, 1)
            pyplot.title('Cross Entropy Loss')
            pyplot.plot(history[i].history['loss'], color='blue', label='train')
            pyplot.plot(history[i].history['val_loss'], color='orange', label='test')
            # plot accuracy
            pyplot.subplot(2, 1, 2)
            pyplot.title('Classification Accuracy')
            pyplot.plot(history[i].history['accuracy'], color='blue', label='train')
            pyplot.plot(history[i].history['val_accuracy'], color='orange', label='test')
        pyplot.show()

        # print summary
        #print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
        # box and whisker plots of results
        pyplot.boxplot(scores)
        pyplot.show()
        

################### Main ###################

if __name__ == "__main__":
    pass