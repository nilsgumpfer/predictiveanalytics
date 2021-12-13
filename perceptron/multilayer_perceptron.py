import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def plot_training_data(X, Y):
    plt.scatter(x=X[:, 0], y=X[:, 1], c=Y, cmap='bwr')
    plt.show()


def plot_training_data_and_activations(X, Y, activations):
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
    plt.tight_layout()
    plt.show()


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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class MultiLayerPerceptron:
    def __init__(self, epochs=1000, learning_rate=0.2, input_neurons=2, hidden_neurons=2, output_neurons=1):
        self.epochs = epochs
        self.lr = learning_rate
        self.hidden_weights = np.random.uniform(size=(input_neurons, hidden_neurons))
        self.hidden_bias = np.random.uniform(size=(1, hidden_neurons))
        self.output_weights = np.random.uniform(size=(hidden_neurons, output_neurons))
        self.output_bias = np.random.uniform(size=(1, output_neurons))

    def predict(self, inputs):
        y_pred, _ = self._forwardpass(inputs)

        return y_pred

    def _forwardpass(self, inputs):
        hidden_layer_activation = np.dot(inputs, self.hidden_weights)
        hidden_layer_activation += self.hidden_bias
        hidden_layer_output = sigmoid(hidden_layer_activation)

        output_layer_activation = np.dot(hidden_layer_output, self.output_weights)
        output_layer_activation += self.output_bias
        predicted_output = sigmoid(output_layer_activation)

        return predicted_output, hidden_layer_output

    def train(self, inputs, outputs):
        # Training algorithm
        for _ in range(self.epochs):
            # Forward pass
            predicted_output, hidden_layer_output = self._forwardpass(inputs)

            # Error calculation
            error = outputs - predicted_output

            # Calculation of derivatives and partial errors
            d_predicted_output = error * sigmoid_derivative(predicted_output)
            error_hidden_layer = d_predicted_output.dot(self.output_weights.T)
            d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

            # Updating Weights and Biases
            self.output_weights += hidden_layer_output.T.dot(d_predicted_output) * self.lr
            self.output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * self.lr
            self.hidden_weights += inputs.T.dot(d_hidden_layer) * self.lr
            self.hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * self.lr


def train(label, epochs, learning_rate):
    np.random.seed(1)
    validation_inputs = [np.array([1, 1]), np.array([1, 0]), np.array([0, 1]), np.array([0, 0])]
    training_inputs, labels = generate_data(100, label)
    labels = labels.reshape((-1, 1))

    # plot_training_data(training_inputs, labels)

    mlp = MultiLayerPerceptron(epochs=epochs, learning_rate=learning_rate)
    mlp.train(training_inputs, labels)

    print('\n')
    for x in validation_inputs:
        print(x, '-->', mlp.predict([x])[0])

    predictions = []

    for x in training_inputs:
        predictions.append(mlp.predict([x])[0])

    plot_training_data_and_activations(training_inputs, labels, predictions)


# Examples from slides
train(label='XOR', epochs=1000, learning_rate=0.2)
train(label='AND', epochs=1000, learning_rate=0.2)
train(label='OR', epochs=1000, learning_rate=0.2)
