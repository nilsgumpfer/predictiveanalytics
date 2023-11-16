import math
import os
from math import tanh

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn.metrics import mean_squared_error


def plot_training_data(X, Y):
    plt.scatter(x=X[:, 0], y=X[:, 1], c=Y, cmap='bwr')
    plt.axhline(y=0.5, color='grey', linestyle='--')
    plt.axvline(x=0.5, color='grey', linestyle='--')
    plt.show()
    plt.close()


def plot_training_data_and_activations(X, Y, activations, epoch=None, save_to=None):
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
    ax.view_init(elev=1, azim=-45)

    if epoch:
        ax.set_title('Epoch {}'.format(epoch))

    if save_to is None:
        plt.show()
    else:
        plt.savefig(save_to)

    plt.close()

    return save_to


def plot_error_gradient(W1, W2, E, final_w1, final_w2, final_e):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs=W1, ys=W2, zs=E, c=E, cmap='viridis')
    ax.scatter(xs=final_w1, ys=final_w2, zs=final_e, c='r', s=300)
    ax.set_xlabel('w1')
    ax.set_ylabel('w2')
    ax.set_zlabel('Error')
    plt.show()
    plt.close()


def plot_error_gradient_course(W1, W2, E, epoch, w1_course, w2_course, save_to=None):
    fig = plt.figure(figsize=(8.5, 7))
    ax = fig.add_subplot()

    ax.tricontour(W1, W2, E, levels=100, linewidths=0.5, colors='k')
    cntr = ax.tricontourf(W1, W2, E, levels=100, cmap="viridis")
    ax.plot(w1_course[:epoch], w2_course[:epoch], c='r', lw=1)
    fig.colorbar(cntr, label='error')
    ax.set_xlabel('w1')
    ax.set_ylabel('w2')

    ax.set_title('Epoch {}'.format(epoch))

    plt.tight_layout()

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

        X.append([x1/100, x2/100])

    return np.asarray(X), np.asarray(Y)


class Perceptron(object):
    def __init__(self, no_of_inputs=2, epochs=200, learning_rate=0.01, learning_rate_decay=0.0, activation_function='binary', bias=True, weight_init=0.0):
        self.history_predictions = []
        # self.history_weights = np.zeros((no_of_inputs, epochs + 1))
        self.history_weights = np.zeros((no_of_inputs + 1, epochs + 1))
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.weights = np.ones(no_of_inputs + 1) * weight_init
        self.activation_function = activation_function
        self.bias = bias
        self.no_of_inputs = no_of_inputs

    def predict(self, inputs):
        if self.bias:
            summation = np.dot(inputs, self.weights[1:]) + self.weights[0] * 1
        else:
            summation = np.dot(inputs, self.weights[1:])

        return self._activation(summation)

    def train(self, training_inputs, labels):
        self.history_weights[..., 0] = self.weights
        self.history_predictions.append([self.predict(inputs) for inputs in training_inputs])

        for e in range(self.epochs):
            predictions_tmp = []
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = (label - prediction)
                d_error = error * self._activation_derivative(prediction)
                self.weights[1:] += self.learning_rate * d_error * inputs
                if self.bias:
                    self.weights[0] += self.learning_rate * d_error * 1
                predictions_tmp.append(self.predict(inputs))

            self.history_predictions.append(predictions_tmp)
            self.history_weights[..., e+1] = self.weights
            self.learning_rate -= self.learning_rate_decay

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


def generate_gif_from_plots(prefix, paths, params, cleanup=False):
    images = []

    for filename in paths:
        images.append(imageio.imread(filename))

    imageio.mimsave('plots/{}_{}.gif'.format(prefix, params), images, loop=0)

    if cleanup:
        for x in paths:
            os.remove(x)


def plot_gradient_descend(perceptron, label, activation_function, epochs, learning_rate, learning_rate_decay, weight_init, weights1, weights2, errors, bias):
    os.makedirs('plots/frames/', exist_ok=True)

    params = '{}_{}_{}_{}_{}_{}_b{}'.format(label, activation_function, epochs, str(learning_rate).replace('.', '-'), str(learning_rate_decay).replace('.', '-'), weight_init, bias)

    paths = []

    for epoch in range(epochs + 1):
        if bias:
            E = np.array(errors[epoch])
        else:
            E = np.array(errors)
        paths.append(plot_error_gradient_course(W1=np.array(weights1), W2=np.array(weights2), E=E, w1_course=perceptron.history_weights[1], w2_course=perceptron.history_weights[2], epoch=epoch, save_to='plots/frames/gradientdescent_{}_{}.png'.format(params, epoch)))

    generate_gif_from_plots('gradientdescent', paths, params, cleanup=True)


def plot_training_data_and_activations_over_time(perceptron, label, activation_function, epochs, learning_rate, learning_rate_decay, weight_init, inputs, labels, bias):
    os.makedirs('plots/frames/', exist_ok=True)

    params = '{}_{}_{}_{}_{}_{}_b{}'.format(label, activation_function, epochs, str(learning_rate).replace('.', '-'), str(learning_rate_decay).replace('.', '-'), weight_init, bias)

    paths = []

    for epoch, p in zip(range(epochs + 1), perceptron.history_predictions):
        paths.append(plot_training_data_and_activations(inputs, labels, p, epoch=epoch, save_to='plots/frames/activations_{}_{}.png'.format(params, epoch)))

    generate_gif_from_plots('activations', paths, params, cleanup=True)


def train(label, activation_function, epochs, learning_rate, weight_init, bias=True, learning_rate_decay=0.0, interactive=True, precision=150, padding=0.2):
    np.random.seed(1)
    validation_inputs = [np.array([1, 1]), np.array([1, 0.01]), np.array([0.01, 1]), np.array([0.01, 0.01])]
    training_inputs, labels = generate_data(100, label)

    if interactive:
        plot_training_data(training_inputs, labels)

    perceptron = Perceptron(activation_function=activation_function, epochs=epochs, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay, bias=bias, weight_init=weight_init)
    perceptron.train(training_inputs, labels)

    print('\n')
    for x in validation_inputs:
        print(x, '-->', perceptron.predict(x))

    predictions = []

    for x in training_inputs:
        predictions.append(perceptron.predict(x))

    final_e = mean_squared_error(labels, predictions)
    final_wb = perceptron.weights[0]
    final_w1 = perceptron.weights[1]
    final_w2 = perceptron.weights[2]

    print('Learned weights: w1={:.4f}, w2={:.4f}, wb={:.4f}, Final error (MSE): {:.4f}'.format(final_w1, final_w2, final_wb, final_e))

    if interactive:
        plot_training_data_and_activations(training_inputs, labels, predictions)

    weights1 = []
    weights2 = []
    errors = []

    w1_low = np.min([weight_init, np.min(perceptron.history_weights[1])]) - padding
    w1_high = np.max([weight_init, np.max(perceptron.history_weights[1])]) + padding
    w2_low = np.min([weight_init, np.min(perceptron.history_weights[2])]) - padding
    w2_high = np.max([weight_init, np.max(perceptron.history_weights[2])]) + padding

    for w1 in np.linspace(start=w1_low, stop=w1_high, num=precision):
        for w2 in np.linspace(start=w2_low, stop=w2_high, num=precision):
            perceptron.weights[1] = w1
            perceptron.weights[2] = w2
            weights1.append(w1)
            weights2.append(w2)

            predictions_tmp = []

            for x, y in zip(training_inputs, labels):
                predictions_tmp.append(perceptron.predict(x))

            e = mean_squared_error(labels, predictions_tmp)
            errors.append(e)

    if interactive:
        plot_error_gradient(weights1, weights2, errors, final_w1=final_w1, final_w2=final_w2, final_e=final_e)

    if bias and not interactive:
        errors_bias_specific = np.zeros((epochs + 1, precision**2))

        for i, wb in enumerate(perceptron.history_weights[0]):
            errors_tmp = []

            for w1 in np.linspace(start=w1_low, stop=w1_high, num=precision):
                for w2 in np.linspace(start=w2_low, stop=w2_high, num=precision):
                    perceptron.weights[0] = wb
                    perceptron.weights[1] = w1
                    perceptron.weights[2] = w2

                    predictions_tmp = []

                    for x, y in zip(training_inputs, labels):
                        predictions_tmp.append(perceptron.predict(x))

                    e = mean_squared_error(labels, predictions_tmp)
                    errors_tmp.append(e)

            errors_bias_specific[i] = np.array(errors_tmp)

        errors = errors_bias_specific

    if not interactive:
        plot_gradient_descend(perceptron, label, activation_function, epochs, learning_rate, learning_rate_decay, weight_init, weights1, weights2, errors, bias)

    plot_training_data_and_activations_over_time(perceptron, label, activation_function, epochs, learning_rate, learning_rate_decay, weight_init, training_inputs, labels, bias)


# Basic examples
# train(label='AND', activation_function='binary', epochs=50, learning_rate=0.001, weight_init=0.0, interactive=True, precision=150, padding=0.2)
# train(label='OR', activation_function='binary', epochs=50, learning_rate=0.001, weight_init=0.0, interactive=True, precision=150, padding=0.2)
# train(label='XOR', activation_function='binary', epochs=50, learning_rate=0.001, weight_init=0.0, interactive=True, precision=150, padding=0.2)
# train(label='AND', activation_function='sigmoid', epochs=100, learning_rate=0.1, weight_init=0.0, interactive=True, precision=50, padding=2)
# train(label='OR', activation_function='sigmoid', epochs=100, learning_rate=0.1, weight_init=0.0, interactive=True, precision=50, padding=2)
# train(label='XOR', activation_function='sigmoid', epochs=100, learning_rate=0.1, weight_init=0.0, interactive=True, precision=50, padding=2)
# train(label='AND', activation_function='relu', epochs=50, learning_rate=0.01, weight_init=0.0, interactive=True, precision=50, padding=2)
# train(label='OR', activation_function='relu', epochs=50, learning_rate=0.01, weight_init=0.0, interactive=True, precision=50, padding=2)
# train(label='XOR', activation_function='relu', epochs=50, learning_rate=0.01, weight_init=0.0, interactive=True, precision=50, padding=2)

# Advanced examples:
# train(label='AND', activation_function='sigmoid', epochs=100, learning_rate=0.1, weight_init=0, interactive=False, precision=50, padding=0.5)
# train(label='AND', activation_function='sigmoid', epochs=100, learning_rate=0.1, weight_init=2.5, interactive=False, precision=50, padding=0.5)
# train(label='AND', activation_function='sigmoid', epochs=100, learning_rate=0.01, weight_init=0, interactive=False, precision=50, padding=0.5)
# train(label='AND', activation_function='sigmoid', epochs=100, learning_rate=0.01, weight_init=2.5, interactive=False, precision=50, padding=0.5)
# train(label='AND', activation_function='relu', epochs=50, learning_rate=0.001, weight_init=2.5, interactive=False, bias=False, precision=50, padding=0.5)
# train(label='AND', activation_function='relu', epochs=50, learning_rate=0.01, weight_init=2.5, interactive=False, bias=False, precision=50, padding=0.5)
# train(label='AND', activation_function='relu', epochs=50, learning_rate=0.001, weight_init=2.5, interactive=False, precision=50, padding=0.5)
# train(label='AND', activation_function='relu', epochs=50, learning_rate=0.01, weight_init=2.5, interactive=False, precision=50, padding=0.5)
# train(label='XOR', activation_function='sigmoid', epochs=100, learning_rate=0.1, weight_init=0.0, interactive=False, precision=50, padding=2)
# train(label='XOR', activation_function='relu', epochs=50, learning_rate=0.01, weight_init=0.0, interactive=False, precision=50, padding=2)
# train(label='XOR', activation_function='sigmoid', epochs=100, learning_rate=0.1, weight_init=2.5, interactive=False, precision=50, padding=2)
# train(label='XOR', activation_function='relu', epochs=50, learning_rate=0.01, weight_init=2.5, interactive=False, precision=50, padding=2)

# TODO: insert to MD (top)
train(label='AND', activation_function='binary', epochs=50, learning_rate=0.001, weight_init=1, interactive=False, precision=80, padding=0.2)
train(label='AND', activation_function='binary', epochs=50, learning_rate=0.01, weight_init=1, interactive=False, precision=80, padding=0.2)
train(label='AND', activation_function='binary', epochs=50, learning_rate=0.1, weight_init=1, interactive=False, precision=80, padding=0.2)
train(label='AND', activation_function='binary', epochs=50, learning_rate=0.001, weight_init=0.1, interactive=False, precision=80, padding=0.2)
train(label='AND', activation_function='binary', epochs=50, learning_rate=0.01, weight_init=0.1, interactive=False, precision=80, padding=0.2)
train(label='AND', activation_function='binary', epochs=50, learning_rate=0.1, weight_init=0.1, interactive=False, precision=80, padding=0.2)
train(label='AND', activation_function='binary', epochs=50, learning_rate=0.001, weight_init=0, interactive=False, precision=80, padding=0.2)
train(label='AND', activation_function='binary', epochs=50, learning_rate=0.01, weight_init=0, interactive=False, precision=80, padding=0.2)
train(label='AND', activation_function='binary', epochs=50, learning_rate=0.1, weight_init=0, interactive=False, precision=80, padding=0.2)




