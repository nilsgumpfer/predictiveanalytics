import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


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


def plot_input_variants(inputs, inputs_shifted, inputs_scaled):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    ax.set_title('Input value variants')
    ax.scatter(x=inputs[..., 0], y=inputs[..., 1], label='raw', s=10)
    ax.scatter(x=inputs_shifted[..., 0], y=inputs_shifted[..., 1], label='shifted', s=10)
    ax.scatter(x=inputs_scaled[..., 0], y=inputs_scaled[..., 1], label='scaled', s=10)
    ax.axhline(y=0, color='grey', linestyle='--')
    ax.axvline(x=0, color='grey', linestyle='--')
    ax.set_xlim((-3, 8))
    ax.set_ylim((-3, 8))
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


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


def train(epochs=20, learning_rate=0.1, preprocessing=None):
    # Set random seed for reproducibility
    np.random.seed(1)

    # Load iris dataset
    iris = load_iris()
    inputs = iris['data']
    labels = iris['target']

    # Print features, labels, and shapes
    print(iris['feature_names'])
    print(iris['target_names'])
    print(np.shape(inputs), np.shape(labels))
    print(labels)

    # Prepare inputs: reshape labels, select records (first 100 = classes 0 and 1)
    labels = labels.reshape((-1, 1))
    inputs = inputs[:100, 0:2]
    labels = labels[:100]

    # Preprocess inputs
    inputs_scaled = StandardScaler().fit_transform(inputs)
    inputs_shifted = np.zeros_like(inputs)
    inputs_shifted[..., 0] = inputs[..., 0] - np.min(inputs[..., 0])
    inputs_shifted[..., 1] = inputs[..., 1] - np.min(inputs[..., 1])

    # Plot input variants
    plot_input_variants(inputs, inputs_shifted, inputs_scaled)

    # Select input variant to be used
    if preprocessing == 'shift':
        inputs_preprocessed = inputs_shifted
    elif preprocessing == 'scale':
        inputs_preprocessed = inputs_scaled
    else:
        inputs_preprocessed = inputs

    # Train MLP
    mlp = MultiLayerPerceptron(epochs=epochs, learning_rate=learning_rate)
    mlp.train(inputs_preprocessed, labels)

    # Predict using model
    predictions = []
    for x in inputs_preprocessed:
        predictions.append(mlp.predict([x])[0][0])

    # Plot activations
    plot_training_data_and_activations(inputs, labels, predictions)


# First trials, 20 epochs
train(epochs=20, learning_rate=0.1)
# train(epochs=20, learning_rate=0.1, preprocessing='scale')
# train(epochs=20, learning_rate=0.1, preprocessing='shift')

# Further trials, more epochs
# train(epochs=100, learning_rate=0.05)
# train(epochs=200, learning_rate=0.05)
# train(epochs=100, learning_rate=0.1, preprocessing='scale')
# train(epochs=100, learning_rate=0.1, preprocessing='shift')
