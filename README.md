# Automated Machine Learning Using Genetic Algorithm
This project proposes a nested optimization algorithm for automated machine learning. The algorithm optimizes
both the architecture as well as the hyperparameters of the neural network using:
1) A Nelder-Mead algorithm, optimizing the architecture and parameters of the genetic algorithm
2) A genetic algorithm, training a fixed neural network architecture

As an exemplary problem set, the MNIST dataset is used here.

It is possible to specify the complexity of the resulting neural networks by giving depth and width constraints to the algorithm.
The algorithm is capable of training both convolutional neural networks and multi-layer perceptrons (see folder */MLP*).