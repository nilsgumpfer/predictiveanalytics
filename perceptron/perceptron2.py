import math
import os
from math import tanh

import imageio as imageio
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn.metrics import mean_squared_error


def plot_training_data_and_activations(epoch, X, Y, e, activations, azim=-39, elev=8, save_to=None):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs=X[:, 0], ys=X[:, 1], zs=activations, c=Y, cmap='bwr')
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlim([0, 1])
    ax.set_xlabel('x1')
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylim([0, 1])
    ax.set_ylabel('x2')
    ax.set_zticks([0, 0.5, 1])
    ax.set_zlim([0, 1])
    ax.set_zlabel('Activation')

    ax.view_init(azim=azim, elev=elev)
    ax.set_title('Epoch {}, Error: {:.3f}'.format(epoch, e))

    if save_to is None:
        plt.show()
    else:
        plt.savefig(save_to)

    plt.close()

    return save_to

def plot_error_gradient(epoch, W1, W2, E, w1, w2, e, azim=99, elev=22, save_to=None):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs=W1, ys=W2, zs=E, c=E, cmap='viridis')
    ax.scatter(xs=w1, ys=w2, zs=e, c='r', s=300)
    ax.set_xlabel('w1')
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_ylabel('w2')
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_zlim([0, 1])
    ax.set_zticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_zlabel('Error')
    ax.view_init(azim=azim, elev=elev)
    ax.set_title('Epoch {}, Error: {:.3f}'.format(epoch, e))

    if save_to is None:
        plt.show()
    else:
        plt.savefig(save_to)

    plt.close()

    return save_to


def generate_data(n, label):
    X = []
    Y = []

    for x1, x2 in zip(np.random.randint(low=0, high=100, size=n), np.random.randint(low=0, high=100, size=n)):
        if label == 'AND':
            if x1 >= 50 and x2 >= 50:
                Y.append(1)
            else:
                Y.append(0)
        if label == 'OR':
            if x1 >= 50 or x2 >= 50:
                Y.append(1)
            else:
                Y.append(0)
        if label == 'XOR':
            if (x1 >= 50 > x2) or (x1 < 50 <= x2):
                Y.append(1)
            else:
                Y.append(0)

        X.append([x1 / 100, x2 / 100])

    return np.asarray(X), np.asarray(Y)


class Perceptron2(object):
    def __init__(self, no_of_inputs=2, epochs=200, learning_rate=0.01, activation_function='binary', bias=True, weight_init=0.0):
        self.history_predictions = []
        self.epochs = epochs
        self.no_of_inputs = no_of_inputs
        self.learning_rate = learning_rate
        self.weights = np.ones(no_of_inputs + 1) * weight_init
        self.activation_function = activation_function
        self.bias = bias
        self.history_error = np.zeros((epochs + 1,))
        self.history_weights = np.zeros((no_of_inputs, epochs + 1))

    def predict(self, inputs):
        if self.bias:
            summation = np.dot(inputs, self.weights[1:]) + self.weights[0] * 1
        else:
            summation = np.dot(inputs, self.weights[1:])

        return self._activation(summation)

    def train(self, training_inputs, labels):
        for e in range(self.epochs + 1):
            predictions = []

            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                predictions.append(prediction)
                error = (label - prediction)
                d_error = error * self._activation_derivative(prediction)

                if e > 0:
                    self.weights[1:] += self.learning_rate * d_error * inputs
                    # self.weights[0] += self.learning_rate * d_error * 1
                    if self.bias:
                        self.weights[0] += self.learning_rate * d_error * 1

            self.history_error[e] = mean_squared_error(labels, predictions)
            self.history_predictions.append(predictions)

            for i in range(self.no_of_inputs):
                self.history_weights[i][e] = self.weights[i + 1]

    def _activation(self, summation):
        if self.activation_function == 'binary':
            if summation > 0:
                return 1
            else:
                return 0
        if self.activation_function == 'identity':
            return summation
        if self.activation_function == 'relu':
            if summation > 0:
                return summation
            else:
                return 0
        if self.activation_function == 'tanh':
            return tanh(summation)
        if self.activation_function == 'sigmoid':
            return 1 / (1 + math.e ** -summation)
        if self.activation_function == 'softplus':
            return math.log((1 + math.e ** summation))

    def _activation_derivative(self, output):
        if self.activation_function == 'binary':
            return 1
            # Mathematically, the derivative of the binary activation function is always 0. The initial definition
            # of the perceptron by Rosenblatt did not consider derivatives as used for other activation functions.
            # Therefore, we simply return 1 in this case to enable a valid weight update.
        if self.activation_function == 'identity':
            return 1
        if self.activation_function == 'relu':
            if output < 0:
                return 0
            else:
                return 1
        if self.activation_function == 'tanh':
            return 1 - output ** 2
        if self.activation_function == 'sigmoid':
            return output * (1 - output)
        if self.activation_function == 'softplus':
            return 1 / (1 + math.e ** -output)


def generate_gif_from_plots(prefix, paths, params):
    images = []

    for filename in paths:
        images.append(imageio.imread(filename))

    imageio.mimsave('plots/{}_{}.gif'.format(prefix, params), images)


def gradient(label, activation_function, epochs, learning_rate, weight_init, bias=False):
    np.random.seed(1)
    training_inputs, labels = generate_data(100, label)

    perceptron = Perceptron2(activation_function=activation_function, bias=bias, no_of_inputs=2, epochs=epochs, learning_rate=learning_rate)

    weights1 = []
    weights2 = []
    errors = []

    for w1 in np.arange(start=-1, stop=1, step=0.05):
        for w2 in np.arange(start=-1, stop=1, step=0.05):
            perceptron.weights[1] = w1
            perceptron.weights[2] = w2

            predictions_tmp = []

            for x, y in zip(training_inputs, labels):
                predictions_tmp.append(perceptron.predict(x))

            e = mean_squared_error(labels, predictions_tmp)

            weights1.append(w1)
            weights2.append(w2)
            errors.append(e)

    if bias:
        perceptron.weights[0] = weight_init

    perceptron.weights[1] = weight_init
    perceptron.weights[2] = weight_init
    perceptron.train(training_inputs, labels)

    # Azimuth and elevation specifiy camera perspective. To generate a cinematic camera swing around the plot,
    # we change azimuth and elevation in each plot. For this, n=epochs values are generated for each parameter.
    # To adapt the start and endpoints, simply adapt start and stop values below.
    azims = np.linspace(start=90, stop=180, num=epochs)
    elevs = np.linspace(start=10, stop=50, num=epochs)

    paths_graddesc = []
    paths_activ = []

    os.makedirs('plots/frames/', exist_ok=True)

    params = '{}_{}_{}_{}_{}{}'.format(label, activation_function, epochs, str(learning_rate).replace('.', '-'), weight_init, {True: '_bias', False: ''}[bias])

    for epoch, w1, w2, e, p, azim, elev in zip(range(epochs + 1), perceptron.history_weights[0], perceptron.history_weights[1], perceptron.history_error, perceptron.history_predictions, azims, elevs):
        if not bias:
            # Only plot error gradient if no bias was used
            paths_graddesc.append(plot_error_gradient(epoch, weights1, weights2, errors, w1=w1, w2=w2, e=e, azim=azim, elev=elev, save_to='plots/frames/gradientdescent_{}_{}.png'.format(params, epoch)))
        else:
            paths_activ.append(plot_training_data_and_activations(epoch, training_inputs, labels, e, p, save_to='plots/frames/activations_{}_{}.png'.format(params, epoch)))

    # Based on the generated plots, GIF images are generated
    if not bias:
        # Only plot error gradient if no bias was used
        generate_gif_from_plots('gradientdescent', paths_graddesc, params)
    else:
        generate_gif_from_plots('activations', paths_activ, params)


# Animated activations, Slides 52-62
# gradient(label='AND', activation_function='sigmoid', epochs=200, learning_rate=0.1, weight_init=0.0, bias=True)
# gradient(label='OR', activation_function='sigmoid', epochs=200, learning_rate=0.1, weight_init=0.0, bias=True)
# gradient(label='XOR', activation_function='sigmoid', epochs=200, learning_rate=0.1, weight_init=0.0, bias=True)
# exit()
# Animated error gradients, Slides 65-73
# gradient(label='AND', activation_function='binary', epochs=50, learning_rate=0.001, weight_init=0.0)
# gradient(label='OR', activation_function='binary', epochs=50, learning_rate=0.001, weight_init=0.0)
# gradient(label='XOR', activation_function='binary', epochs=50, learning_rate=0.001, weight_init=0.0)
# gradient(label='AND', activation_function='sigmoid', epochs=50, learning_rate=0.1, weight_init=0.0)
# gradient(label='OR', activation_function='sigmoid', epochs=50, learning_rate=0.1, weight_init=0.0)
# gradient(label='XOR', activation_function='sigmoid', epochs=50, learning_rate=0.1, weight_init=0.0)
# gradient(label='AND', activation_function='relu', epochs=50, learning_rate=0.001, weight_init=0.0)
# gradient(label='OR', activation_function='relu', epochs=50, learning_rate=0.001, weight_init=0.0)
# gradient(label='XOR', activation_function='relu', epochs=50, learning_rate=0.001, weight_init=0.0)

# Animated error gradients with different start points -> analyze path, variate epochs and learning rate
# gradient(label='AND', activation_function='binary', epochs=50, learning_rate=0.001, weight_init=-1.0)
# gradient(label='OR', activation_function='binary', epochs=50, learning_rate=0.001, weight_init=-1.0)
# gradient(label='XOR', activation_function='binary', epochs=50, learning_rate=0.001, weight_init=-1.0)
gradient(label='AND', activation_function='sigmoid', epochs=50, learning_rate=0.1, weight_init=-1.0)
gradient(label='OR', activation_function='sigmoid', epochs=50, learning_rate=0.1, weight_init=-1.0)
gradient(label='XOR', activation_function='sigmoid', epochs=50, learning_rate=0.1, weight_init=-1.0)
# gradient(label='AND', activation_function='relu', epochs=50, learning_rate=0.001, weight_init=-1.0)
# gradient(label='AND', activation_function='relu', epochs=100, learning_rate=0.001, weight_init=-1.0)
# gradient(label='OR', activation_function='relu', epochs=50, learning_rate=0.001, weight_init=-1.0)
# gradient(label='XOR', activation_function='relu', epochs=50, learning_rate=0.001, weight_init=-1.0)
# gradient(label='AND', activation_function='binary', epochs=50, learning_rate=0.001, weight_init=1.0)
# gradient(label='OR', activation_function='binary', epochs=50, learning_rate=0.001, weight_init=1.0)
# gradient(label='XOR', activation_function='binary', epochs=50, learning_rate=0.001, weight_init=1.0)
gradient(label='AND', activation_function='sigmoid', epochs=50, learning_rate=0.1, weight_init=1.0)
gradient(label='OR', activation_function='sigmoid', epochs=50, learning_rate=0.1, weight_init=1.0)
gradient(label='XOR', activation_function='sigmoid', epochs=50, learning_rate=0.1, weight_init=1.0)
# gradient(label='AND', activation_function='relu', epochs=50, learning_rate=0.001, weight_init=1.0)
# gradient(label='OR', activation_function='relu', epochs=50, learning_rate=0.001, weight_init=1.0)
# gradient(label='XOR', activation_function='relu', epochs=50, learning_rate=0.001, weight_init=1.0)
