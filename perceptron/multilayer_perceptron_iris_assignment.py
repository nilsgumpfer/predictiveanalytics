import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


def plot_training_data(X, Y):
    plt.scatter(x=X[:, 0], y=X[:, 1], c=Y, cmap='bwr')
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')
    plt.show()


def plot_training_data_and_activations(X, Y, activations):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs=X[:, 0], ys=X[:, 1], zs=activations, c=Y, cmap='bwr')
    ax.set_xlabel('sepal length (cm)')
    ax.set_ylabel('sepal width (cm)')
    ax.set_zticks([0, 0.5, 1])
    ax.set_zlim([0, 1])
    ax.set_zlabel('Activation')
    plt.tight_layout()
    plt.show()


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
    np.random.seed(1)
    inputs, labels = load_iris(return_X_y=True)

    labels = labels.reshape((-1, 1))  # Wrap single-value labels into vector
    inputs = inputs[:100, 0:2]  # Select records of first two classes, of each, select two first features
    labels = labels[:100]  # Select first two classes

    inputs_std_scaled = StandardScaler().fit_transform(inputs)
    inputs_shifted = inputs - 4.5

    fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(15, 10))
    axs[0][0].set_title('Raw input values (scatter)')
    axs[0][0].scatter(y=np.ravel(inputs), x=np.arange(start=0, stop=200))
    axs[0][1].set_title('Raw input values (box-plot)')
    axs[0][1].boxplot(np.ravel(inputs))
    axs[1][0].set_title('Shifted input values (scatter)')
    axs[1][0].scatter(y=np.ravel(inputs_shifted), x=np.arange(start=0, stop=200))
    axs[1][1].set_title('Shifted input values (box-plot)')
    axs[1][1].boxplot(np.ravel(inputs_shifted))
    axs[2][0].set_title('Std-scaled input values (scatter)')
    axs[2][0].scatter(y=np.ravel(inputs_std_scaled), x=np.arange(start=0, stop=200))
    axs[2][1].set_title('Std-scaled input values (box-plot)')
    axs[2][1].boxplot(np.ravel(inputs_std_scaled))
    plt.tight_layout()
    plt.show()
    plt.close()

    # Define which input variant to use
    inputs_preprocessed = inputs_std_scaled

    # Explanation why scaling/shifting/raw values are a good/bad idea
    # Training finally yields usable results, but it may take much less time if the training data is scaled and centered in a "good" way. The longer training can be caused by higher weight values to be learned / adapted during training. It takes much less time to produce low weights, high weight values require much more time to develop.
    #
    # std scaled, 20 epochs, working correctly
    # [[ 1.43618143  1.94041117]
    #  [-1.2114966  -1.47613612]]
    #
    # shifted, 20 epochs, failing
    # [[ 0.70162075  1.39422232]
    #  [-0.1321098   0.04623839]]
    #
    # shifted, 60 epochs, working correctly
    # [[ 1.11354323  1.9798659 ]
    #  [-0.54489349 -1.51883082]]
    #
    # raw, 20 epochs, failing
    # [[ 0.32962256  0.71427478]
    #  [-0.10781781  0.29985366]]
    #
    # raw, 350 epochs, working correctly
    # [[  6.09656861   0.5607521 ]
    #  [-10.22994957   0.41524752]]

    plot_training_data(inputs_preprocessed, labels)

    mlp = MultiLayerPerceptron(epochs=20, learning_rate=0.1)
    mlp.train(inputs_preprocessed, labels)

    print(mlp.hidden_weights, mlp.output_weights)

    predictions = []

    for x in inputs_preprocessed:
        predictions.append(mlp.predict([x])[0][0])

    plot_training_data_and_activations(inputs_preprocessed, labels, predictions)


train()
