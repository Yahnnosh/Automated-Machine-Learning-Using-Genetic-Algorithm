# Automated Machine Learning Using Genetic Algorithm

# Introduction
### Motivation
The goal of this project is the development of a fully automated optimization algorithm for
(convolutional) neural networks. The algorithm optimizes both the neural architecture as well
as all hyperparameters of the network. The following pipeline for the (nested) algorithm is proposed:

### Pipeline
1. A Nelder-Mead algorithm first optimizes the neural architecture and the parameters of the
   ensuing genetic algorithm used for training the network.
2. A genetic algorithm then trains the hyperparameters of a fixed neural network architecture
   from the Nelder-Mead algorithm.

### Problem set
As an exemplary problem set, the MNIST dataset for image classification is used in this project.

# Methods
### Genetic algorithm
The genetic algorithm optimizes all hyperparameters of a (convolutional) neural networks, i.e.:
1. **Weights** of the fully connected layers
2. **Biases** of the fully connected layers
3. **Filters** of the convolutional layers
4. **Activation functions** of all layers (from a list of possible functions to choose)

The genetic algorithm itself uses the following operations:
1. **Initialization**: Fixed population size N
2. **Fitness function**: Accuracy of the neural network
3. **Selection**: Roulette-wheel selection with elitism
4. **Crossover**: Neuron-wise crossover with a fixed crossover probability
5. **Mutation**: Additive gaussian noise with fixed mutation probabilities

### Nelder-Mead algorithm
The Nelder-Mead algorithm optimizes the neural architecture as well as the parameters of the 
genetic algorithm, i.e.:
1. Population size
2. Selection size
3. Mutation probabilities 
4. Crossover probability
5. Gaussian noise variance
6. Convolutional filter dimensions
7. Convolutional filter strides
8. Convolutional pooling strides
9. Fully connected layer widths

# Results
The developed optimization algorithm achieved 87% accuracy on the MNIST dataset on a multi-layer perceptron
and 89% accuracy on a convolutional neural network during testing. Furthermore, the current
algorithm struggles with larger models due to the large amount of optimization parameters
and converges consistently to relatively simple models. Further work is required for the algorithm
to investigate deeper models and profit from the increased performance they can provide.
