import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn.model_selection import train_test_split

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.training.adam import AdamOptimizer


def generate_data():
    X_cls1 = np.concatenate((np.random.normal(-0.5, 0.2, size=(50, 2)), np.random.normal(0.5, 0.2, size=(50, 2))))
    X_cls2 = np.concatenate((np.random.normal(-0.5, 0.2, size=(50, 2)), np.random.normal(0.5, 0.2, size=(50, 2))))
    X_cls2[..., 1] = X_cls2[..., 1] * -1

    X = np.concatenate((X_cls1, X_cls2))

    Y = np.concatenate((np.ones(50), np.ones(50), np.zeros(50), np.zeros(50)))

    return X, Y


def plot_training_data(X, Y):
    plt.scatter(x=X[:, 0], y=X[:, 1], c=Y, cmap='bwr')
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
        Dense(4, activation='relu', input_shape=(2,)),
        Dense(1, activation='sigmoid')
    ])

    # Compile model
    model.compile(optimizer=AdamOptimizer(learning_rate=0.01),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train model
    history = model.fit(X, y, epochs=250, validation_data=(X_test, y_test))

    print('Accuracy:', max(history.history['accuracy']))

    return model(X).numpy()


def train():
    np.random.seed(1)
    X, Y = generate_data()
    Y = Y.reshape((-1, 1))

    pred = train_MLP(X, Y)

    plot_training_data_and_activations(X, Y, pred)


train()
