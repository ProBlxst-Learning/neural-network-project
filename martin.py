

import keras
from matplotlib import pyplot


VERBOSE = True


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
        self.__nn.add(keras.layers.core.Dense(int(input_size),
                                              input_dim=int(input_size), activation='relu'))

        # make one layer per hidden layer with the specified number of neurons
        [self.__nn.add(keras.layers.core.Dense(neurons, activation='relu'))
         for neurons in hidden_layers],

        # the output layer shoud be as large as the defined output size
        self.__nn.add(keras.layers.core.Dense(
            output_size, activation='softmax'))

        # compile model
        self.__nn.compile(optimizer=keras.optimizers.SGD(
            lr=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

        print(self.__nn.summary())

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
        capacity = self.__layers[0]*self.__layers[1]  # weights
        capacity += self.__layers[1]  # biases
        capacities.append(capacity)
        print('bit capacity layer %s: %s' % (1, capacity)) if VERBOSE else None

        # for all but the first layer the capacity contribution is min(rule 2, output of previous layer) this is rule 3
        for i in range(2, len(self.__layers)):
            print(i)
            capacity = self.__layers[i-1]*self.__layers[i]  # weights
            capacity += self.__layers[i]  # biases
            capacities.append(min(capacity, self.__layers[i-1]))
            print('bit capacity layer %s: %s' %
                  (i, capacities[i-1])) if VERBOSE else None

        return sum(capacities)

    # used to train network with given training data

    def evaluate(self, train_x, train_y, test_x, test_y, epochs=5):

        # we wish to se the progress the model went through in the training, both in traing performed and the accuracy it achieved
        fit = 1

        print('Training initiated') if VERBOSE else None

        # train the model and record the proces
        print('fit initiated') if VERBOSE else None
        history = self.__nn.fit(train_x, train_y, epochs=epochs, batch_size=32, validation_data=(
            test_x, test_y), verbose=VERBOSE)
        print('fit finished') if VERBOSE else None

        # for every fit call, print the accuracy and recort it
        print('evaluate initiated') if VERBOSE else None
        loss, acc = self.__nn.evaluate(test_x, test_y, verbose=1)
        print('evaluate finished') if VERBOSE else None
        print(acc)
        print('>%s: %.3f' % (fit, acc))
        print('Loss', loss)
        fit += 1
        score = acc

        # the training needs to be added to the history of the model
        self.__histories = history
        self.__scores = score

        # the model is now trained and the diagnistics should be returned
        return history, score

    # should visualise the training

    def visualise_training(self, history=None, scores=None):

        history = history if history else self.__histories
        scores = scores if scores else self.__scores

        print(history)
        print(history.history)

        # plot loss
        pyplot.subplot(2, 1, 1)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(history.history['loss'],
                    color='blue', label='train')
        pyplot.plot(history.history['val_loss'],
                    color='orange', label='test')
        # plot accuracy
        pyplot.subplot(2, 1, 2)
        pyplot.title('Classification Accuracy')
        pyplot.plot(history.history['accuracy'],
                    color='blue', label='train')
        pyplot.plot(history.history['val_accuracy'],
                    color='orange', label='test')
        pyplot.show()

        # print summary
        #print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
        # box and whisker plots of results
        # pyplot.boxplot(scores)
        # pyplot.show()


def load_dataset():

    (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()

    return train_x, train_y, test_x, test_y

# the input data needs to be a float32 between 0 and 1 and the output data needs to be a categorical


def format_data(data_x, data_y):

    # convert input data to float32 and normalise
    data_x_float = data_x.astype('float32')
    input_norm = data_x_float/255.0

    # flatten pictures to one dimention
    input_flatt = input_norm.reshape((len(data_x), 28*28))

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
    model = NN(28*28, 10, 16)

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


main()
