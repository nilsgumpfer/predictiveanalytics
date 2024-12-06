import os

import matplotlib.pyplot as plt
import numpy as np
from flask import session
from keras.applications.vgg16 import decode_predictions
from signxai.methods.wrappers import calculate_relevancemap
from signxai.utils.utils import load_image, aggregate_and_normalize_relevancemap_rgb
from tensorflow.keras.applications.vgg16 import VGG16

# Load model
model = VGG16(weights='imagenet')

#  Remove last layer's softmax activation (we need the raw values!)
model.layers[-1].activation = None

neuron_selection = 3

path = '../data/'
for filename in os.listdir(path):
    # Load example image
    # img, x = load_image('../data/zebra-4.jpg')
    # img, x = load_image('../data/zebra-6.jpg')
    # img, x = load_image('../data/zebra-8.jpg')
    # img, x = load_image('../data/zebra-9.jpg')
    # img, x = load_image('../data/zebra-14.jpg')
    # img, x = load_image('../data/zebra-15.jpeg')
    # img, x = load_image('../data/zebra-18.jpg')
    # img, x = load_image('../data/zebra-19.jpg')

    if not (filename.endswith('png') or filename.endswith('jpg') or filename.endswith('jpeg')):
        continue

    if not filename.startswith('shark'):
        continue

    img, x = load_image(path + filename)
    print(filename)

    pred = model.predict(np.array([x]))
    print(decode_predictions(pred, top=3)[0])

    # Calculate relevancemaps
    R1 = aggregate_and_normalize_relevancemap_rgb(calculate_relevancemap('gradient', np.array(x), model, neuron_selection=neuron_selection))
    R2 = aggregate_and_normalize_relevancemap_rgb(calculate_relevancemap('gradient_x_input', np.array(x), model, neuron_selection=neuron_selection))
    R3 = aggregate_and_normalize_relevancemap_rgb(calculate_relevancemap('gradient_x_sign', np.array(x), model, neuron_selection=neuron_selection))

    img1 = np.array(img)
    img1[R1 > 0.06, 0] = 255
    img1[R1 > 0.06, 1] = 255 - (R1[R1 > 0.06] * 255)
    img1[R1 > 0.06, 2] = 255 - (R1[R1 > 0.06] * 255)

    img2 = np.array(img)
    img2[R2 > 0.06, 0] = 255
    img2[R2 > 0.06, 1] = 255 - (R2[R2 > 0.06] * 255)
    img2[R2 > 0.06, 2] = 255 - (R2[R2 > 0.06] * 255)

    img3 = np.array(img)
    img3[R3 > 0.06, 0] = 255
    img3[R3 > 0.06, 1] = 255 - (R3[R3 > 0.06] * 255)
    img3[R3 > 0.06, 2] = 255 - (R3[R3 > 0.06] * 255)

    # Visualize heatmaps
    fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(24, 12))
    axs[0][0].imshow(img)
    axs[0][1].matshow(R1, cmap='seismic', clim=(-1, 1))
    axs[0][1].set_title('Gradient')
    axs[0][2].matshow(R2, cmap='seismic', clim=(-1, 1))
    axs[0][2].set_title(r'Gradient $\times$ Input')
    axs[0][3].matshow(R3, cmap='seismic', clim=(-1, 1))
    axs[0][3].set_title(r'Gradient $\times$ SIGN')

    axs[1][0].imshow(img)
    axs[1][1].imshow(img1)
    axs[1][2].imshow(img2)
    axs[1][3].imshow(img3)

    plt.tight_layout()

    plt.savefig('../data/plots/' + filename + '_{}.png'.format(neuron_selection))
