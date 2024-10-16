import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist


def main():
    # Load data
    (train_images, train_labels), (val_images, val_labels) = mnist.load_data()

    # TODO: classify digits


if __name__ == '__main__':
    main()

