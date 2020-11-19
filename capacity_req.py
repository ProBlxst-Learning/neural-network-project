
import math
import numpy as np
import tensorflow as tf

from tqdm import tqdm


def upload_and_return_set(_type):

    dataset = getattr(tf.keras.datasets, _type)

    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    # Only training data is needed for heuristic
    return x_train, y_train


def max_cap_req(data, labels, _type):
    '''
    Calculate the maximum and approximated expected capacity reuirement of a binary classifier neural network for a given training data
    :param data: List filled with a 2D-list each representing one data point. Each element is
    :param labels: List of length i with
    :param _type: Need to behaave a bit different if it's CIFAR or MNIST, related to dimensions. Includes randomized for our random list
    '''

    # thresholds is iterated each time labels does a switch in the sorted table
    thresholds = 0

    # length of the dataset and number of dimensions in each vector
    length = len(data)
    d = len(data[0])**2
    # different dimensions for cifar
    if _type == 'cifar10':
        d = 32*32*3
    # initilization of table with the same length as the data and zero's before loop
    table = [[0, 0] for _ in range(length)]

    # Table will loop through the data and summarize each dimension in the vector
    if _type == 'cifar10':
        for i in range(length):
            mapsum = 0
            for j in range(len(data[i])):
                mapsum += sum(map(sum, data[i][j]))
            table[i] = [mapsum, labels[i]]
    else:
        for i in range(length):
            table[i] = [sum(map(sum, data[i])), labels[i]]

    # Table is sorted on the first key, and _class is set to zero
    sorted_table = sorted(table)
    _class = 0

    # For each time the label in the new sorted table does not match the _class, _class is updated with the item in
    # the sorted_table, and threshold is increased with one
    for i in range(length):
        if not sorted_table[i][1] == _class:
            _class = sorted_table[i][1]
            thresholds += 1

    # Calculation of maximum capacity requirement and expected capacity requirement
    _max_cap_req = thresholds * d + thresholds + 1
    _exp_cap_req = math.log2(thresholds + 1) * d

    return thresholds, _max_cap_req, _exp_cap_req


def pretty_output(table, datasets):
    """
    Takes in a 2d-array with the goal of printing the results with a pretty output
    """
    print("_"*50)
    print("OVERVIEW OF RESULTS FROM CAPACITY ESTIMATOR".center(50))

    for i in range(len(table)):
        print("_"*50)
        print(f"Output from {datasets[i]}: ".center(50))
        print("_"*50)
        for j in range(len(table[0])):
            if j == 0:
                print("Thresholds " + "|".rjust(21) +
                      str(table[i][j]).rjust(5))
            elif j == 1:
                print("Maximum capacity requirement " +
                      "|".rjust(3) + str(table[i][j]).rjust(8))
            else:
                print("Expected capacity requirement " + "|".rjust(2) +
                      str(round(table[i][j], 3)).rjust(7))

    print("_"*50)
    print("METHOD COMPLETED".center(50))
    print("_"*50)


def main():
    datasets = ["mnist", "fashion_mnist", "cifar10"]

    table = []
    for set in datasets:
        print("Beginning dataset: ", set)
        data, labels = upload_and_return_set(set)
        print("Data loaded ... beginning capacity estimator")

        thresholds, _max_cap_req, _exp_cap_req = max_cap_req(data, labels, set)
        print("Method done. Results: ")
        print(thresholds, _max_cap_req, _exp_cap_req)
        table.append([thresholds, _max_cap_req, _exp_cap_req])
        print("Results appended to table: \n", table)

    pretty_output(table, datasets)


if __name__ == '__main__':
    main()
