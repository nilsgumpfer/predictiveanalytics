import matplotlib.cm as cm
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


def aggregate_and_normalize_heatmaps_rgb(h):
    # Aggregate along color channels and normalize to [-1, 1]
    a = h.sum(axis=3)
    a = normalize_heatmap(a)

    return a


def normalize_heatmap(h):
    # Normalize to [-1, 1]
    a = h / np.max(np.abs(h))
    a = np.nan_to_num(a, nan=0)

    return a


def make_grad_heatmap(img_array, model):
    ins = tf.convert_to_tensor(img_array)

    with tf.GradientTape() as tape:
        tape.watch(ins)
        preds = model(ins)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to each input pixel
    grad = tape.gradient(class_channel, ins)

    return grad.numpy()


def save_and_display_grad(img_path, heatmap, cam_path, cmap_name, alpha):
    heatmap = heatmap + 1
    heatmap = heatmap / 2

    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use colormap to colorize heatmap
    cmap = cm.get_cmap(cmap_name)

    # Use RGB values of the colormap
    cmap_colors = cmap(np.arange(256))[:, :3]
    cmap_heatmap = cmap_colors[heatmap]

    # Create an image with RGB colorized heatmap
    cmap_heatmap = keras.preprocessing.image.array_to_img(cmap_heatmap)
    cmap_heatmap = cmap_heatmap.resize((img.shape[1], img.shape[0]))
    cmap_heatmap = keras.preprocessing.image.img_to_array(cmap_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = cmap_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)


def main(cmap_name='seismic', alpha=2.0):
    # Load model
    model = ResNet152(weights='imagenet')

    # Parameters
    img_path = '../data/rooster.jpg'
    # img_path = '../data/myrooster.jpg'
    # img_path = '../data/tower.jpg'
    # img_path = '../data/castle.jpg'
    # img_path = '../data/castledark.jpg'
    # img_path = '../data/elephant.jpg'
    # img_path = '../data/giraffe.jpg'
    cam_path = '{}_vanillagrad.jpg'.format(img_path.rsplit('.', maxsplit=1)[0])

    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Remove last layer's softmax activation (we need the raw values!)
    model.layers[-1].activation = None

    # Calculate gradient
    heatmap = make_grad_heatmap(x, model)

    # Aggregate and normalize heatmap
    heatmap_a_n = aggregate_and_normalize_heatmaps_rgb(heatmap)[0]

    # Display heatmap
    plt.imshow(heatmap_a_n, cmap=cmap_name, clim=(-1, 1))
    plt.show()

    # Create and save superimposed visualization
    save_and_display_grad(img_path, heatmap_a_n, cam_path, cmap_name, alpha)


main()
