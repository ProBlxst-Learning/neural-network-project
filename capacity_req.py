

import random
import math


# Function we can use to create a distributed random list to test two classes
def create_distributed_random_list():
    data, labels = [], []

    for i in range(50):
        labels.append(random.randint(0, 1))
        class_0 = random.uniform(0, 0.75)
        class_1 = random.uniform(0.25, 1)
        if labels[i] == 0:
            data.append(class_0)
        else:
            data.append(class_1)
    return data, labels


# Creating a random list of 0 and 1
def create_random_list():
    return [random.randint(0, 1) for i in range(50)]


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
    if _type == 'cifar':
        d = 32*32*3
    if _type == 'randomized':
        d = 28
    # initilization of table with the same length as the data and zero's before loop
    table = [[0, 0] for _ in range(length)]

    # Table will loop through the data and summarize each dimension in the vector
    if _type == 'cifar':
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

    print("Total amount of threshold: " + str(thresholds))

    # Calculation of maximum capacity requirement and expected capacity requirement
    _max_cap_req = thresholds * d + thresholds + 1
    _exp_cap_req = math.log2(thresholds + 1) * d

    print("Max: " + str(_max_cap_req) + " bits")
    print("Exp: " + str(round(_exp_cap_req, 3)) + " bits")


data, labels = create_distributed_random_list()

max_cap_req(data, labels, 'randomized')
