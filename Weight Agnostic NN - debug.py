import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models, Model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import random
import time


# %%
def softmax(x):
    z = x - np.max(x)  # overflow protection (softmax(x) = softmax(x - const))
    return np.exp(z) / np.sum(np.exp(z))


# TODO: change
activation_functions = {
    'tanh': tf.tanh,
    'relu': tf.nn.relu,
    'sigmoid': tf.nn.sigmoid,
    'linear': (lambda x: x),
    'softmax': tf.nn.softmax
}
# %%
t0 = time.time()
# numpy
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784).astype(np.float32) / 255.0
X_test = X_test.reshape(10000, 784).astype(np.float32) / 255.0

y_train = to_categorical(y_train)  # one-hot encoding
y_test = to_categorical(y_test)  # one-hot encoding

# tensorflow
X_test = tf.convert_to_tensor(X_test)
Y_test = tf.convert_to_tensor(y_test)
# %%
MUTATE_RATE_MATRIX = 0.3
MUTATE_RATE_BIAS = 0.1
MUTATE_RATE_ACTIVATION_FUNCTION = 0.1
GAUSSIAN_NOISE_STDDEV = 1


# %%
# TODO: good idea?
def bitstring_mutation(param, mutate_rate, n_bits=7):
    try:
        # TODO: check if correct (esp. sign)
        bitstring = bin(param)[2:].zfill(n_bits)
        temp = ''
        sign = +1 if random.uniform(0, 1) else -1
        for i in range(len(bitstring)):
            if random.uniform(0, 1) < mutate_rate:
                temp2 = '0' if bitstring[i] == '1' else '1'
                temp += temp2
            else:
                temp += bitstring[i]
        return sign * int(temp, 2)

    except (TypeError):
        bitstring = bin(param)[2:].zfill(n_bits)
        temp = ''
        sign = +1 if random.uniform(0, 1) else -1
        for i in range(len(bitstring)):
            if random.uniform(0, 1) < mutate_rate:
                temp2 = '0' if bitstring[i] == '1' else '1'
                temp += temp2
            else:
                temp += bitstring[i]
        print(param, temp)


# %%
# ---- DEBUG from here ----
class LinModel(Model):
    def __init__(self, matrix1, bias1):
        """
        Weight gnostic multi-layer feed forward neural network
        :param params: Params have to be in form: (matrix1=..., bias1=..., activation1=..., matrix2=..., ...)
        """
        super(LinModel, self).__init__()

        self.linear1 = tf.keras.layers.Dense(32,
                                             activation='sigmoid',
                                             kernel_initializer=tf.keras.initializers.Constant(matrix1),
                                             bias_initializer=tf.keras.initializers.Constant(bias1),
                                             name='linear1')
        '''self.linear2 = tf.keras.layers.Dense(10,
                                             activation='softmax',
                                             kernel_initializer=tf.keras.initializers.Constant(matrix2),
                                             bias_initializer=tf.keras.initializers.Constant(bias2))'''

    def call(self, inputs):
        x = self.linear1(inputs)
        '''x = self.linear2(x)'''
        return x


# %%
bias1 = tf.random.normal(mean=0.0, stddev=1.0, shape=(32,))
matrix1 = tf.random.normal(mean=0.0, stddev=1.0, shape=(784, 32))
bias2 = tf.random.normal(mean=0.0, stddev=1.0, shape=[10, 1])
matrix2 = tf.random.normal(mean=0.0, stddev=1.0, shape=[10, 32])

lin = LinModel(matrix1=matrix1, bias1=bias1)
lin.compile(metrics=['accuracy'])
lin.evaluate(X_test, Y_test)[1]
#%%
bias1 = tf.random.normal(mean=0.0, stddev=1.0, shape=[32, 1])
matrix1 = tf.random.normal(mean=0.0, stddev=1.0, shape=[32, 784])
bias2 = tf.random.normal(mean=0.0, stddev=1.0, shape=[10, 1])
matrix2 = tf.random.normal(mean=0.0, stddev=1.0, shape=[10, 32])

temp = MultiLayerPerceptron(matrix1=matrix1, bias1=bias1, activation1='sigmoid', matrix2=matrix2, bias2=bias2, activation2='softmax')
#%%
%%timeit
temp.evaluate()
# ---- DEBUG until here ----
# %%
class MultiLayerPerceptron(Model):
    # TODO: variable layers
    def __init__(self, **params):
        """
        Weight gnostic multi-layer feed forward neural network
        :param params: Params have to be in form: (matrix1=..., bias1=..., activation1=..., matrix2=..., ...)
        """
        super(MultiLayerPerceptron, self).__init__()

        for (param_name, param) in params.items():
            setattr(self, param_name, param)

    def call(self, input):
        x = input

        i = 1
        while hasattr(self, 'matrix' + str(i)):
            x = getattr(self, 'matrix' + str(i)) @ x
            x += getattr(self, 'bias' + str(i))
            x = activation_functions[getattr(self, 'activation' + str(i))](x)

            i += 1

        return x

    def evaluate(self):
        y_pred = np.argmax(self.call(X), axis=0)
        y_true = np.argmax(Y, axis=0)
        return np.mean(y_pred == y_true)

    # TODO: CHECK IF ABOVE WORKS THEN CHANGE BELOW!!!!!!!!!!!!!!

    '''def mutate(self):
        # TODO: quite arbitrary, inefficient and do negative numbers work?
        # TODO: only works for integers as of now

        # connectivity matrix
        for matrix_name in ('matrix1', 'matrix2'):
            matrix = getattr(self, matrix_name)
            # TODO: clamp to min, max?
            mutation_stencil = tf.cast(tf.reshape(tf.random.categorical(
                tf.math.log([[1 - MUTATE_RATE_MATRIX, MUTATE_RATE_MATRIX]]),
                matrix.shape[0] * matrix.shape[1]), matrix.shape), tf.float32)
            noise = tf.random.normal(mean=0.0, stddev=GAUSSIAN_NOISE_STDDEV, shape=matrix.shape)  # TODO: tune stddev
            matrix = matrix + tf.multiply(mutation_stencil, noise)
            setattr(self, matrix_name, matrix)

        # bias
        for bias_name in ('bias1', 'bias2'):
            bias = getattr(self, bias_name)
            mutation_stencil = tf.cast(tf.reshape(tf.random.categorical(
                tf.math.log([[1 - MUTATE_RATE_BIAS, MUTATE_RATE_BIAS]]),
                bias.shape[0] * bias.shape[1]), bias.shape), tf.float32)
            noise = tf.random.normal(mean=0.0, stddev=GAUSSIAN_NOISE_STDDEV, shape=bias.shape)  # TODO: tune stddev
            bias = bias + tf.multiply(mutation_stencil, noise)
            setattr(self, bias_name, bias)

        # activation function
        # TODO: keep softmax at the end?
        for activation_name in ('activation1'):
            if random.uniform(0, 1) < MUTATE_RATE_ACTIVATION_FUNCTION:
                activation = random.choice(list(activation_functions.keys()))
                setattr(self, activation_name, activation)'''


# %%
class Population():
    def __init__(self, size=10, n_survivors=5):
        self.generation = 0
        self.size = size
        self.n_survivors = n_survivors
        self.elite = None

        # initialization (gaussian)
        # TODO: max, min for now 7-bit integers
        self.organisms = []
        for _ in range(size):
            # TODO: for now fixed architecture
            bias1 = tf.random.normal(mean=0.0, stddev=1.0, shape=[32, 1])
            matrix1 = tf.random.normal(mean=0.0, stddev=1.0, shape=[32, 784])
            activation1 = 'sigmoid'

            bias2 = tf.random.normal(mean=0.0, stddev=1.0, shape=[10, 1])
            matrix2 = tf.random.normal(mean=0.0, stddev=1.0, shape=[10, 32])
            activation2 = 'softmax'

            model = MultiLayerPerceptron(matrix1, bias1, activation1, matrix2, bias2, activation2)
            model.compile()

            self.organisms.append(model)

        self.history = [
            (max(self.organism_fitness()), self.average_fitness())]  # fitness of population over all generations

    def organism_fitness(self):
        return [organism.evaluate() for organism in self.organisms]

    def average_fitness(self):
        organism_fitness = self.organism_fitness()
        return sum(organism_fitness) / len(organism_fitness)

    def max_fitness(self):
        return max(self.organism_fitness())

    def selection(self):
        organism_fitness = self.organism_fitness()

        # elitism (n=1)
        elite_index = np.argmax(organism_fitness)
        self.elite = self.organisms.pop(elite_index)
        organism_fitness.pop(elite_index)

        probabilities = [fitness / sum(organism_fitness) for fitness in organism_fitness]  # normalized
        survivors = np.random.choice(self.organisms,
                                     size=self.n_survivors - 1,
                                     p=probabilities,
                                     replace=False)  # TODO: works without replacement and p?
        return [survivor for survivor in survivors]

    def crossover(self, parents):
        # TODO: for different type of networks
        # TODO: correct?
        children = []
        while len(children) < (self.size - 1):
            [father, mother] = random.sample(parents + [self.elite], k=2)  # sample without replacement

            # TODO: for now assume same no of layers
            # TODO: create new model - efficient?

            # bias, activation function (full gene crossover) # TODO: good?
            child_bias1 = father.bias1 if (random.uniform(0, 1) < 0.5) else mother.bias1
            child_bias2 = father.bias2 if (random.uniform(0, 1) < 0.5) else mother.bias2

            child_activation1 = father.activation1 if (random.uniform(0, 1) < 0.5) else mother.activation1
            child_activation2 = father.activation2 if (random.uniform(0, 1) < 0.5) else mother.activation2

            # matrix (uniform (bit-wise) crossover) # TODO: good?
            father_stencil = tf.round(tf.random.uniform(father.matrix1.shape))
            mother_stencil = - (father_stencil - 1)
            child_matrix1 = tf.multiply(father_stencil, father.matrix1) + tf.multiply(mother_stencil, mother.matrix1)

            father_stencil = tf.round(tf.random.uniform(father.matrix2.shape))
            mother_stencil = - (father_stencil - 1)
            child_matrix2 = tf.multiply(father_stencil, father.matrix2) + tf.multiply(mother_stencil, mother.matrix2)

            model = MultiLayerPerceptron(child_matrix1,
                                         child_bias1,
                                         child_activation1,
                                         child_matrix2,
                                         child_bias2,
                                         child_activation2)
            model.compile()  # TODO: necessary??
            children.append(model)

        return children

    def mutate(self, organisms):
        for organism in organisms:
            organism.mutate()

    def breed(self):
        # time_debug = ''

        # t_a = time.time()
        parents = self.selection()
        # t_b = time.time()
        # time_debug += 'selection time: {}s - '.format(round(t_b - t_a, 4))

        # t_a = time.time()
        children = self.crossover(parents)
        # t_b = time.time()
        # time_debug += 'crossover time: {}s - '.format(round(t_b - t_a, 4))

        # t_a = time.time()
        self.mutate(children)  # TODO: mGA or GA?
        # t_b = time.time()
        # time_debug += 'mutation time: {}s - '.format(round(t_b - t_a, 4))

        # print(time_debug)

        self.organisms = children + [self.elite]
        self.generation += 1
        self.history.append((self.max_fitness(), self.average_fitness()))

    def plot(self):
        plt.figure()
        plt.plot(np.arange(self.generation + 1), [score[0] for score in self.history],
                 label='max fitness')
        plt.plot(np.arange(self.generation + 1), [score[1] for score in self.history],
                 label='avg fitness', alpha=0.6)
        plt.title('Population fitness' + ' (n=' + str(self.size) + ')')
        plt.xlabel('Generations')
        plt.ylabel('Fitness score (accuracy)')
        plt.legend()
        plt.show()


# %%
# initialization
GENERATIONS = 1000
POPULATION_SIZE = 100
SURVIVORS = 35
population = Population(size=POPULATION_SIZE, n_survivors=SURVIVORS)
# %%
# initial population
print('Starting training')
t_training = time.time()
population_fitness = population.organism_fitness()
max_fitness = population.max_fitness()
t2 = time.time()
print('Gen', 0, ':',
      population_fitness, '- max:',
      max_fitness,
      '({}s)'.format(round(t2 - t_training, 2)))

# future populations
for generation in range(1, GENERATIONS):
    # breed new population
    t1 = time.time()
    population.breed()

    # evaluate new population
    population_fitness = population.organism_fitness()
    max_fitness = population.max_fitness()
    t2 = time.time()

    print('Gen', generation, ':',
          population_fitness, '- max:',
          max_fitness,
          '({}s)'.format(round(t2 - t1, 2)))

print('Finished training ({})'.format(round(time.time() - t_training, 2)))
print('\nTotal computation time: ({}s)'.format(round(time.time() - t0, 2)))

# performance of population
population.plot()
