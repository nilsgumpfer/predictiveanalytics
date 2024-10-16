import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

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


def train():
    # Set random seed for reproducibility
    np.random.seed(1)

    # Load iris dataset
    iris = load_iris()
    inputs = iris['data']
    labels = iris['target']

    # TODO: implement preprocessing and training


train()
