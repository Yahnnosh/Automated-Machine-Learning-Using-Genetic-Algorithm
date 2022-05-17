# Automatic Machine Learning Using Genetic Algorithm
This project proposes an algorithm used for training a neural network (architecture and hyperparameters), evaluated here on MNIST and CIFAR-100.
A genetic algorithm trains a fixed neural network architecture, while a second algorithm (Nelder-mead?) optimizes the architecture and the parameters used in the genetic algorithm.

# Files
**MNIST**
- genetic_algorithm_MNIST.py: offers run function, returning accuracy of best performing network on MNIST
- genetic_algorithm_MNIST_notebook.ipynb: interactive genetic algorithm on MNIST

**CIFAR-100**
- genetic_algorithm_CIFAR-100_py: offers run function, returning accuracy of best performing network on CIFAR-100
- genetic_algorithm_CIFAR-100_notebook.ipynb: interactive genetic algorithm on CIFAR-100

# Mind Map
- ~~_(Much) faster evaluation -> computation time_~~
- ~~_Multi-layer NN_~~
- ~~_Weight gnostic NN?_~~
- _Number of layers mutable -> speciation_
- _Convolutional layers_
- ~~_Lots of activation functions_~~
- _Activation function by neuron_
- ~~_Row wise (neuron wise) crossover_~~
- _Dynamic mutation rate_
- _Dynamic population size_
- ~~_Crossover rate_~~
- ~~_Subset (of X_test) evaluation_~~ -> effect negligable
- _GA parameter tuning_ -> e.g. bayesian optimization, Nelder-mead?
