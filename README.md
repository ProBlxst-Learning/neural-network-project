# Neural-network-project
Repository containing code for the group project in the course TDT4173 - Machine Learning, Fall 2020. This README focuses on how our results can be reproduced from others. It also described links between the project paper and the codes found in this repository.

## Getting started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You must make sure you have python and pip installed on your computer.

Run `python3 --version` in your terminal to see if you have python installed.
If not, you can go to [python.org](https://www.python.org/downloads/) to install the latest version of python.

Run `pip --version` to see if you have pip installed. pip is installed with python.

You will later need to use a virtual environment. 
Run `pip install virtualenv` to make sure you have this installed. 

### Installing

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

* model - the neural network models, where dense initializes the model and the three other trains on datasets
  * `dense.py`
  * `mnist.py`
  * `fashion_mnist.py`
  * `cifar10.py`
* utils - contains the method for calculating capacity requirements
   * `capacity_req.py`
* img - stores images
  * ... .png

## Running the files

Make sure you have virtualenv activated and installed everything in requirements.txt. If not, follow description in "installing"

### Capacity estimator
From root write `cd utils`
Run the capacity estimator with `python3 capacity_req.py`
It loads and iterates through three datasets, so it may take some minutes. Print statements makes sure the process can be followed through terminal. At then end, the output is printed to terminal to be used for creation of neural networks.

### Training and running neural networks
From root write `cd models`
* Train and evaluate MNIST: `python3 mnist.py`
* Train and evaluate Fashion MNIST: `python3 fashion_mnist.py`
* Train and evaluate CIFAR-10: `python3 cifar10.py`

The neural network uses keras, so the process can be followed from the terminal.

## Connection with report

### Method for capacity requirements

The method can be found in `utils/capacity_req.py`. It loads all three datasets and iterates through all in order to find the data we need. The result is shown below, and is the same as can be found in the report in section 5.1.

<img src=https://github.com/ProBlxst-Learning/neural-network-project/blob/main/img/capacity_req_output.png width="350" />

The next sections are related to the neural networks. The three python-files `mnist.py`, `fashion_mnist.py` and `cifar10.py` is structured in the same way, but loads different datasets in the beginning. Unless other specified, the descriptions beneath are related to these files. 

### Data
The data is loaded at the beginning of the file. The function `load_dataset()` in the file loads the data in variables for later use, splitting the data in training and test at initialization. It uses keras to load the data, found in `keras.datasets` supplied with the dataset and the keras-function `load_data()`. An overview of these dataset is found in section 3.1 in the report.

### Preprocessing
Preprocessing is the next thing after data is loaded. This happens in the function `format_data(data_x, data_y)`. MNIST and Fashion MNIST reshapes the input data in 28x28 - dimensions, while CIFAR10 use 32x32x3. This pre-processing is further specified in section 3.2 in the report.

### Neural network initialization
Three networks of different sizes are initialized using the class NN from `dense.py`. It specifices input data neurons and output neurons as required arguments, while the hidden layers are not required. This shows how we can have one neural net with no hidden layers while the next two with one hidden layer. The layer structure and related bit capacity of this is shown in section 5.2 in the report.

### Training and evaluation

Training and evaluation is done in the function `train` in `dense.py`. The result stores the accuracy on the test dataset. The test accuracy from each of the different neural networks sizes are plotted on the same graph with the function `compare_training`, also found in `dense.py`. These graphs is shown in section 5.3 in the report.

