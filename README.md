# Neural-network-project
Repository containing code for the group project in the course TDT4173 - Machine Learning, Fall 2020. The project works in two parts: It implements a heuristic for determining the size of a neural network from a given dataset. Subsequently, neural networks are created and trained on the respective daatasets.

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

Use `git clone` to clone the repo.

Enter the directory with `cd neural-network-project`.

Create a virtual environment, where we use python 3.8.6.

Python version needs to be >3.6.0 to get all installments in the right way.

Create the virtual environment with python 3.8 using `virtualenv -p python3.8 venv`

Activate the virtualenv for
- Mac users: `source venv/bin/activate`
- Windows users: `venv\Scripts\activate`

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

The three next sections is the same for the different python-files each representing a dataset, found in model.
### Data
The data is loaded at the beginning of the python-file for the different datasets. The function `load_dataset()`uses keras to load the data. With keras installed, this data will be able to load locally.

### Preprocessing
Preprocessing is the next thing after data is loaded. This happens in the function `format_data(data_x, data_y)`.

### Neural network initialization
Three networks of different sizes are initialized using the class NN from `dense.py`.

### Training and evaluation

Training and evaluation is done in the function `train()` in `dense.py`. The result is accuracy on the test dataset, shown in section 5.3 in the report. Note: Need to run each dataset independently. 

### Capacity method

The method can be found in `utils/capacity_req.py`. It loads all three datasets and iterates through all in order to find the data we need. The result is shown below, and is the same as can be found in the report in section 5.1.

![Output from capacity estimator](https://github.com/ProBlxst-Learning/neural-network-project/blob/main/img/capacity_req_output.png =200x)
