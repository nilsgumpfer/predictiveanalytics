import math
import numpy as np
import matplotlib.pyplot as plt


def activation(x, activation_function):
    if activation_function == 'binary':
        if x > 0:
            return 1
        else:
            return 0
    if activation_function == 'identity':
        return x
    if activation_function == 'relu':
        if x > 0:
            return x
        else:
            return 0
    if activation_function == 'tanh':
        return math.tanh(x)
    if activation_function == 'sigmoid':
        return 1 / (1 + math.e ** -x)
    if activation_function == 'softplus':
        return math.log((1 + math.e ** x))


def activation_derivative(x, activation_function):
    if activation_function == 'binary':
        return 1
        # Mathematically, the derivative of the binary activation function is always 0. The initial definition
        # of the perceptron by Rosenblatt did not consider derivatives as used for other activation functions.
        # Therefore, we simply return 1 in this case to enable a valid weight update.
    if activation_function == 'identity':
        return 1
    if activation_function == 'relu':
        if x < 0:
            return 0
        else:
            return 1
    if activation_function == 'tanh':
        return 1 - x ** 2
    if activation_function == 'sigmoid':
        return x * (1 - x)
    if activation_function == 'softplus':
        return 1 / (1 + math.e ** -x)


def main(activation_function='relu'):
    X = np.arange(start=-2, stop=2, step=0.1)
    Y = [activation(x, activation_function) for x in X]

    w = 0.5
    xi = 2
    y = 0.1
    lr = 0.5

    for _ in range(20):
        plt.plot(X, Y)

        y_hat = activation(w * xi, activation_function)
        print('y_hat', y_hat)
        plt.plot(w * xi, y_hat, 'or')

        error = y - y_hat
        w = w + lr * error * activation_derivative(y_hat, activation_function) * xi

        plt.plot(w * xi, activation(w * xi, activation_function), 'og')

        plt.show()
        plt.close()



main()
