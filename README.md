# Neural-network-project
Repository containing code from TDT4173 - Machine Learning, final project


## Getting started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You must make sure you have python and pip installed on your computer.

Run `python3 --version` in your terminal to see if you have python installed.
If not, you can go to [python.org](https://www.python.org/downloads/) to install the latest version of python.

Run `pip --version` to see if you have pip installed. pip is installed with python.

You will later need to use a virtual environment. 
Run `pip install virtualenv` to make sure you have this installed. 

## Installing

First we got to create a virtual environment, where we use python 3.8.6.

Python version needs to be >3.6.0 to get all installments in the right way. 

`virtualenv -p python3.8 venv`

Activate the virtualenv

Mac users:
`source venv/bin/activate`

Windows users:
`venv\Scripts\activate`

Install the requirements for the project and migrate the database

`pip install -r requirements.txt`


## File structure

* model - the neural network models, both for initialization and results
  * `dense.py` - creates a class, NN, which initialises aas a sequential keras model with dense layers. Performs the training in the function name train. Includes functions for calculating bit capacity and visualize the training. 

The next three python files are related to each of the three dataset: MNIST, Fashion MNIST and CIFAR-10. Structured in three different files to decide which dataset to run.
Each contains two functions, load and format the data, as well as main function. The neural net for each datasets uses initiaalization and functions from `dense.py` in the main method.
  * `mnist.py`
  * `fashion_mnist.py`
  * `cifar10.py`

* utils - contains one way, the capacity requirement method
   * `capacity_req.py` - runs through all three datasets and stores the results in a table. Prints a pretty output when run with main-function. 
   
   
## Connection with report

### Data

### Preprocessing


### Neural network initialization


### Capacity method and results

![Output from capacity estimator](https://github.com/ProBlxst-Learning/neural-network-project/blob/main/img/capacity_req_output.png)

### Results


