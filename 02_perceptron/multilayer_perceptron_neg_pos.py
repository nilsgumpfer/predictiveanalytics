import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from signxai.utils.utils import remove_softmax
from sklearn.model_selection import train_test_split

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import save_model, load_model
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.keras.callbacks import ModelCheckpoint
import tensorflow as tf


def generate_data():
    X_cls1 = np.concatenate((np.random.normal(-0.5, 0.2, size=(50, 2)), np.random.normal(0.5, 0.2, size=(50, 2))))
    X_cls2 = np.concatenate((np.random.normal(-0.5, 0.2, size=(50, 2)), np.random.normal(0.5, 0.2, size=(50, 2))))
    X_cls2[..., 1] = X_cls2[..., 1] * -1

    X = np.concatenate((X_cls1, X_cls2))

    X_r = np.zeros((np.shape(X)[0], np.shape(X)[1]+1))
    X_r[..., 0] = X[..., 0]
    X_r[..., 1] = X[..., 1]
    X_r[..., 2] = np.random.normal(0, 1, size=np.shape(X)[0])

    Y = np.concatenate((np.ones(50), np.ones(50), np.zeros(50), np.zeros(50)))

    # return X, Y
    return X_r, Y


def plot_training_data(X, Y):
    plt.scatter(x=X[:, 0], y=X[:, 2], c=Y, cmap='bwr')
    plt.show()


def plot_training_data_and_activations(X, Y, activations):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs=X[:, 0], ys=X[:, 1], zs=activations, c=Y, cmap='bwr')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Activation')
    plt.tight_layout()
    plt.show()



def train_MLP(X, y, random_seed=0):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed, stratify=y)

    # Build model
    model = Sequential([
        Dense(3, activation='relu', input_shape=(np.shape(X)[-1],)),
        Dense(1, activation='sigmoid')
    ])

    # Compile model
    model.compile(optimizer=AdamOptimizer(learning_rate=0.01),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    checkpoint_callback = ModelCheckpoint(filepath='MLP.h5', monitor='val_accuracy', mode='max', save_best_only=True)

    # Train model
    history = model.fit(X, y, epochs=200, validation_data=(X_test, y_test), callbacks=[checkpoint_callback])

    print('Accuracy:', max(history.history['accuracy']))

    return model(X).numpy()


def train():
    np.random.seed(1)
    X, Y = generate_data()
    Y = Y.reshape((-1, 1))

    # plot_training_data(X, Y)
    # plot_training_data_and_activations(X, Y, X[..., 2])

    pred = train_MLP(X, Y)

    plot_training_data_and_activations(X, Y, pred)


def explain():
    # XOR data
    X = np.array([[-1, -1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, 1]])
    X = X / 2
    Y = np.array([1, 0, 0, 1])

    model = load_model('MLP.h5')
    # model = remove_softmax(load_model('MLP.h5'))

    # Select a few samples for explanation
    sample_inputs = tf.convert_to_tensor(X, dtype=tf.float32)

    # Gradient computation
    with tf.GradientTape() as tape:
        tape.watch(sample_inputs)  # Watch the input tensor
        predictions = model(sample_inputs)  # Forward pass

    # Compute the gradients of the first output neuron w.r.t. the inputs
    gradients = tape.gradient(predictions, sample_inputs).numpy()

    print(np.round(gradients, 2))


def main():
    # train()
    explain()



if __name__ == '__main__':
    main()