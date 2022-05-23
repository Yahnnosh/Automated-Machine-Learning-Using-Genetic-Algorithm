import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras import layers, models, Model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import random
import time

activation_functions = {
    'tanh': tf.tanh,
    'relu': tf.nn.relu,
    'sigmoid': tf.nn.sigmoid,
    'linear': tf.keras.activations.linear,
    'softmax': tf.nn.softmax,
    'sign': tf.sign,
    'sin': tf.sin,
    'exp': tf.exp
}


# Loading Data
SUBSET = 1.0  # subset (in percentage) of X_test used during training

# numpy
_, (X_test, y_test) = cifar100.load_data()  # only care  about X_test

selection = np.random.choice(np.arange(X_test.shape[0]),
                             int(SUBSET * X_test.shape[0]),
                             replace=False)

X_test = X_test.reshape(10000, 3072).astype(np.float32)[selection] / 255.0  # flatten
y_test = to_categorical(y_test)[selection]  # one-hot encoding

# tensorflow
X_test = tf.convert_to_tensor(np.transpose(X_test))
y_test = tf.convert_to_tensor(np.transpose(y_test))

# data for evaluation
y_true = np.argmax(y_test, axis=0)


# Network definition
MUTATE_RATE_MATRIX = 0.2
MUTATE_RATE_BIAS = 0.2
MUTATE_RATE_ACTIVATION_FUNCTION = 0.2
CROSSOVER_RATE = 0.5
GAUSSIAN_NOISE_STDDEV = 1  # mutation applies additive gaussian noise
UNIFORM_CROSSOVER = False  # if True, performs crossover of matrices element-wise, else row-wise
HIDDEN_LAYER_WIDTH = 100


class MultiLayerPerceptron(Model):
    def __init__(self, **params):
        """
        Weight gnostic multi-layer feed forward neural network
        :param params: Params have to be in form: (matrix1=..., bias1=..., activations1=..., matrix2=..., ...)
        """
        super(MultiLayerPerceptron, self).__init__()

        self.n_layers = max(
            [int(param_name[-1]) for param_name in params.keys()])  # = number of hidden layers + 1 (output layer)

        for (param_name, param) in params.items():
            assert param_name[:-1] in ('matrix', 'bias', 'activation'), 'Invalid attribute!'
            setattr(self, param_name, param)

    def call(self, inputs):
        x = inputs

        for layer in range(1, self.n_layers + 1):
            x = getattr(self, 'matrix' + str(layer)) @ x
            x += getattr(self, 'bias' + str(layer))
            x = activation_functions[getattr(self, 'activation' + str(layer))](x)

        return x

    def evaluate(self):
        y_pred = np.argmax(self.call(X_test), axis=0)
        return np.mean(y_pred == y_true)

    def mutate(self):
        for layer in range(1, self.n_layers + 1):
            # matrix
            matrix = getattr(self, 'matrix' + str(layer))
            mutation_stencil = tf.cast(tf.reshape(tf.random.categorical(
                tf.math.log([[1 - MUTATE_RATE_MATRIX, MUTATE_RATE_MATRIX]]),
                matrix.shape[0] * matrix.shape[1]), matrix.shape), tf.float32)
            noise = tf.random.normal(mean=0.0, stddev=GAUSSIAN_NOISE_STDDEV, shape=matrix.shape)
            matrix = matrix + tf.multiply(mutation_stencil, noise)
            setattr(self, 'matrix' + str(layer), matrix)

            # bias
            bias = getattr(self, 'bias' + str(layer))
            mutation_stencil = tf.cast(tf.reshape(tf.random.categorical(
                tf.math.log([[1 - MUTATE_RATE_BIAS, MUTATE_RATE_BIAS]]),
                bias.shape[0]), bias.shape), tf.float32)
            noise = tf.random.normal(mean=0.0, stddev=GAUSSIAN_NOISE_STDDEV, shape=bias.shape)
            bias = bias + tf.multiply(mutation_stencil, noise)
            setattr(self, 'bias' + str(layer), bias)

            # activation
            cleaner = lambda x: 'softmax' if x == 'softmax_v2' else x
            activation = cleaner(getattr(self, 'activation' + str(layer)))
            if random.uniform(0, 1) < MUTATE_RATE_ACTIVATION_FUNCTION:
                activation = random.choice(list(activation_functions.keys()))
            setattr(self, 'activation' + str(layer), activation)

    def summary(self):
        dash = '-' * 75
        ddash = '=' * 75
        print(dash)
        print('Model')
        print(ddash)

        n_params = 0
        for layer in range(1, self.n_layers + 1):
            # get values
            matrix = getattr(self, 'matrix' + str(layer))
            bias = getattr(self, 'bias' + str(layer))
            cleaner = lambda x: 'softmax' if x == 'softmax_v2' else x
            activation = cleaner(getattr(self, 'activation' + str(layer)))

            n_params += matrix.shape[0] * matrix.shape[1] + bias.shape[0] + 1

            # print adjustments
            activation = '({})'.format(activation)
            layer_IO = '(in={}, out={})'.format(matrix.shape[1], matrix.shape[0], )

            print('Linear {:<20}{:<30}#Params: {}'.format(activation, layer_IO,
                                                          matrix.shape[0] * matrix.shape[1] + bias.shape[0] + 1))

        print(ddash)
        print('Total params: {}'.format(n_params))
        print('Accuracy: {}\n'.format(round(self.evaluate(), 3)))


class Population:
    def __init__(self, size=10, n_survivors=5, n_hidden_layers=1):
        """
        :param size: population size
        :param n_survivors: number of survivors after each generation (rest is killed and unable to pass on its genes)
        :param n_hidden_layers: number of hidden layers
        """
        self.generation = 0
        self.size = size
        self.n_survivors = n_survivors
        self.n_hidden_layers = n_hidden_layers
        self.elite = None
        self.fitness = None  # cache fitness for increased speed
        self.fitness_generation = -1  # generation when fitness was evaluated

        # initialization (gaussian)
        self.organisms = []
        for _ in range(size):
            params = {}

            n_neurons_prev = 3072
            n_neurons_curr = HIDDEN_LAYER_WIDTH
            for layer in range(1, self.n_hidden_layers + 2):
                if layer == self.n_hidden_layers + 1:
                    n_neurons_curr = 100  # output layer
                params['matrix' + str(layer)] = tf.random.normal(mean=0.0, stddev=1.0,
                                                                 shape=[n_neurons_curr, n_neurons_prev])
                params['bias' + str(layer)] = tf.random.normal(mean=0.0, stddev=1.0, shape=[n_neurons_curr, 1])
                params['activation' + str(layer)] = 'sigmoid'
                n_neurons_prev = HIDDEN_LAYER_WIDTH

            model = MultiLayerPerceptron(**params)
            self.organisms.append(model)

        self.history = [
            (max(self.organism_fitness()), self.average_fitness())]  # fitness of population over all generations

    def organism_fitness(self):
        if self.generation != self.fitness_generation:
            self.fitness = [organism.evaluate() for organism in self.organisms]
            self.fitness_generation = self.generation

        return self.fitness

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
                                     replace=False)
        return [survivor for survivor in survivors]

    def crossover(self, parents):
        children = []
        while len(children) < int(CROSSOVER_RATE * (self.size - 1)):
            [father, mother] = random.sample(parents + [self.elite], k=2)  # sample without replacement

            child_params = {}
            for layer in range(1, father.n_layers + 1):
                if UNIFORM_CROSSOVER:
                    # matrix - uniform crossover
                    father_matrix = getattr(father, 'matrix' + str(layer))
                    mother_matrix = getattr(mother, 'matrix' + str(layer))

                    father_mask = tf.round(tf.random.uniform(father_matrix.shape))
                    mother_mask = - (father_mask - 1)

                    child_matrix = tf.multiply(father_mask, father_matrix) + tf.multiply(mother_mask, mother_matrix)
                    child_params['matrix' + str(layer)] = child_matrix
                else:
                    # matrix - row-wise (neuron-wise) crossover
                    father_matrix = getattr(father, 'matrix' + str(layer))
                    mother_matrix = getattr(mother, 'matrix' + str(layer))

                    n_rows = father_matrix.shape[0]
                    father_mask = np.random.choice([True, False], size=n_rows)

                    child_matrix = tf.convert_to_tensor([father_matrix[row, :] if father_mask[row]
                                                         else mother_matrix[row, :] for row in range(n_rows)])
                    child_params['matrix' + str(layer)] = child_matrix

                # bias - uniform crossover
                father_bias = getattr(father, 'bias' + str(layer))
                mother_bias = getattr(mother, 'bias' + str(layer))

                father_mask = tf.round(tf.random.uniform(father_bias.shape))
                mother_mask = - (father_mask - 1)

                child_bias = tf.multiply(father_mask, father_bias) + tf.multiply(mother_mask, mother_bias)
                child_params['bias' + str(layer)] = child_bias

                # activation
                cleaner = lambda x: 'softmax' if x == 'softmax_v2' else x
                father_activation = cleaner(getattr(father, 'activation' + str(layer)))
                mother_activation = cleaner(getattr(mother, 'activation' + str(layer)))

                child_activation = father_activation if (random.uniform(0, 1) < 0.5) else mother_activation
                child_params['activation' + str(layer)] = child_activation

            model = MultiLayerPerceptron(**child_params)
            children.append(model)

        # if CROSSOVER_RATE != 100% allow some individuals to pass on their genes without crossover
        while len(children) < (self.size - 1):
            [model] = random.sample(parents + [self.elite], k=1)  # sample without replacement

            child_params = {}
            for layer in range(1, model.n_layers + 1):
                # matrix
                child_params['matrix' + str(layer)] = getattr(model, 'matrix' + str(layer))

                # bias
                child_params['bias' + str(layer)] = getattr(model, 'bias' + str(layer))

                # activation
                cleaner = lambda x: 'softmax' if x == 'softmax_v2' else x
                child_params['activation' + str(layer)] = cleaner(getattr(model, 'activation' + str(layer)))

            model = MultiLayerPerceptron(**child_params)
            children.append(model)

        return children

    def mutate(self, organisms):
        for organism in organisms:
            organism.mutate()

    def breed(self, debug=False):
        if debug:
            time_debug = ''

            t_a = time.time()
            parents = self.selection()  # ~0.0005s
            t_b = time.time()
            time_debug += 'selection time: {}s - '.format(round(t_b - t_a, 4))

            t_a = time.time()
            children = self.crossover(parents)  # ~0.28s
            t_b = time.time()
            time_debug += 'crossover time: {}s - '.format(round(t_b - t_a, 4))

            t_a = time.time()
            self.mutate(children)  # ~0.15s#
            t_b = time.time()
            time_debug += 'mutation time: {}s - '.format(round(t_b - t_a, 4))

            print(time_debug)
        else:
            parents = self.selection()
            children = self.crossover(parents)
            self.mutate(children)

        self.organisms = children + [self.elite]
        self.generation += 1
        self.history.append((self.max_fitness(), self.average_fitness()))

    def plot(self):
        # plot evolution
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

        # plot best performing final network
        organism_fitness = self.organism_fitness()
        elite_index = np.argmax(organism_fitness)
        self.organisms[elite_index].summary()


# Training
def run(population_size, survivor_size, generations, hidden_layers, hidden_layer_width, mutation_rate_matrix,
        mutation_rate_bias, mutation_rate_activation_function, crossover_rate, gaussian_noise_stdd=1):
    """
    Runs genetic algorithm on the specified population.
    :param population_size: #individuals during each generation
    :param survivor_size: #individuals that survive BEFORE crossover and mutation
    :param generations: #generations to run genetic algorithm
    :param hidden_layers: #hidden layers of neural network (minimum 0)
    :param hidden_layer_width: #neurons in hidden layers (all the same)
    :param mutation_rate_matrix: probability that gaussian noise is added for each weight
    :param mutation_rate_bias: probability that gaussian noise is added for each bias
    :param mutation_rate_activation_function: probability that activation function is replaced randomly for that layer
    :param crossover_rate: ratio of children that are produced with crossover (rest is drawn from parent generation)
    :param gaussian_noise_stdd: standard deviation of additive gaussian noise in mutation phase
    :return: accuracy of best performing neural network in last generation
    """

    # Set global variables
    global MUTATE_RATE_MATRIX
    global MUTATE_RATE_BIAS
    global MUTATE_RATE_ACTIVATION_FUNCTION
    global CROSSOVER_RATE
    global GAUSSIAN_NOISE_STDDEV
    global HIDDEN_LAYER_WIDTH

    MUTATE_RATE_MATRIX = mutation_rate_matrix
    MUTATE_RATE_BIAS = mutation_rate_bias
    MUTATE_RATE_ACTIVATION_FUNCTION = mutation_rate_activation_function
    CROSSOVER_RATE = crossover_rate
    GAUSSIAN_NOISE_STDDEV = gaussian_noise_stdd
    HIDDEN_LAYER_WIDTH = hidden_layer_width

    # Run algorithm
    population = Population(size=population_size, n_survivors=survivor_size, n_hidden_layers=hidden_layers)

    for generation in range(1, generations + 1):
        population.breed()

    return population.max_fitness()




def to_param(config):
    return (round(config[0]), round(config[1]), round(config[2]), round(config[3]), round(config[4]), config[5], config[6], config[7], config[8], config[9])


def random_config():
    pop = random.randint(5, 20)
    return np.array([pop, random.randint(0, pop), 50, random.randint(0, 3), random.randint(5, 50),
        random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), random.uniform(0.1, 2)])

def assert_bounds(config):
    return config



def downhill(iterations):
    configs = np.zeros((10, 9))
    scores = np.zeros(9)

    print("start")

    for i in range(9):
        config = random_config()
        score = run(*to_param(config))
        print("run: ", i, " random ", score, config)
        configs[:, i] = config
        scores[i] = score
    

    for i in range(9, iterations):
        worst_idx = scores.argmin()
        worst_config = configs[:, worst_idx]
        configs[:, worst_idx] = np.zeros(10)
        scores[worst_idx] = 2.0

        centeroid = np.mean(config, axis=1)
        diff = centeroid - worst_config

        new_config = assert_bounds(centeroid + diff)
        new_score = run(*to_param(new_config))


        if new_score > min(scores):
            configs[:, worst_idx] = new_config
            scores[worst_idx] = new_score
            print("run: ", i, " fullstep ", new_score, new_config)
            continue


        new_config = assert_bounds(centeroid + diff/2)
        new_score = run(*to_param(new_config))
        configs[:, worst_idx] = new_config
        scores[worst_idx] = new_score
        



        







if __name__ == "__main__":
    downhill(15)