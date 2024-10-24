import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist


def main():
    # Load data
    (train_images, train_labels), (val_images, val_labels) = mnist.load_data()

    print(np.shape(train_images))
    print(train_labels[0])

    for c in set(train_labels):
        tmp = train_images[train_labels==c]
        m = np.mean(tmp, axis=0)
        plt.matshow(m, cmap='Greys')
        plt.show()
        plt.close()

    plt.matshow(train_images[0], cmap='Greys')
    plt.show()

    # TODO: implement classification

if __name__ == '__main__':
    main()