import matplotlib.pyplot as plt
import numpy as np
from signxai.methods.wrappers import calculate_relevancemap
from signxai.utils.utils import normalize_heatmap
from tensorflow.keras.datasets import fashion_mnist, mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical


def prepare_data():
    # Load train and test data
    ((train_images, train_labels), (val_images, val_labels)), ds_name = fashion_mnist.load_data(), 'fashion'
    # ((train_images, train_labels), (val_images, val_labels)), ds_name = mnist.load_data(), 'digits'

    # Normalize color values (here: grey-scales)
    train_images = train_images / 255.0
    val_images = val_images / 255.0

    # Expand pixel dimension (1 color channel)
    train_images = np.expand_dims(train_images, axis=3)
    val_images = np.expand_dims(val_images, axis=3)

    # Do one-hot encoding / do categorical conversion
    train_labels = to_categorical(train_labels)
    val_labels = to_categorical(val_labels)

    # Extract number of classes from data dimensions
    nclasses = np.shape(train_labels)[1]

    return train_images, train_labels, val_images, val_labels, nclasses, ds_name


def explain():
    # Load data
    train_images, train_labels, val_images, val_labels, nclasses, ds_name = prepare_data()

    # Load model
    model = load_model('../03_deeplearning/models/mnist_{}_cnn.h5'.format(ds_name))

    # Remove softmax
    model.layers[-1].activation = None

    # Calculate relevancemaps
    np.random.seed(11)
    i = np.random.randint(low=0, high=len(val_images))
    x = val_images[i]
    R1 = calculate_relevancemap('smoothgrad_x_sign', np.array(x), model, mu=0.5)
    R2 = calculate_relevancemap('grad_cam', np.array(x), model, last_conv_layer_name='conv2d_1')
    R1_n = normalize_heatmap(R1)
    R2_n = normalize_heatmap(R2)
    R1_n[R1_n < 0] = 0
    R2_n[R2_n < 0] = 0

    # Visualize heatmaps
    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(12, 4))
    axs[0].imshow(x, cmap='gist_gray_r', clim=(-1, 1))
    axs[0].set_title('input')
    axs[1].matshow(R1_n, cmap='seismic', clim=(-1, 1))
    axs[1].set_title('lrpsign_epsilon_0_1')
    axs[2].matshow(R2_n, cmap='seismic', clim=(-1, 1))
    axs[2].set_title('Grad CAM')
    plt.show()


if __name__ == '__main__':
    explain()
