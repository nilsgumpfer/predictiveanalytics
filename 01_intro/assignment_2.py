import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist


def main(thresh, show_masks=False):
    # Load data
    (train_images, train_labels), (val_images, val_labels) = mnist.load_data()

    # Placeholder dictionary for mean masks
    mean_masks = {}

    # Iterate over all classes
    for c in set(train_labels):

        # Select train images for class c
        v = train_images[train_labels == c]

        # Calculate mean over all images for class c
        m = np.mean(v, axis=0)

        # Apply threshold: set all values below thres to 0, all others to 1
        m[m < thresh] = 0
        m[m >= thresh] = 1

        # Save mask in dictionary
        mean_masks[c] = m

        # In case show_mask is true, show a plot of the mask
        if show_masks:
            plt.matshow(m, cmap='Greys')
            plt.show()
            plt.close()

    # Placeholder variables for accuracy calculation
    correct = 0

    # Iterate over all validation samples
    for x, y in zip(val_images, val_labels):

        # Variables to derive max score
        max_c = -1
        max_score = -1

        # Iterate over all classes in mean_masks
        for c in mean_masks:

            # Multiply mask with sample image
            mult = x * mean_masks[c]

            # Calculate score
            score = np.sum(mult)

            # If score is higher than previous maximum, update max score and class
            if score > max_score:
                max_score = score
                max_c = c

        # If class of sample matches class with maximum score, it's a correct classification
        if y == max_c:
            correct += 1

    # Calculate accuracy
    acc = correct / len(val_images) * 100

    # Print result
    print('Threshold: {}, Accuracy: {:.2f}%'.format(thresh, acc))


if __name__ == '__main__':
    main(thresh=60, show_masks=True)

