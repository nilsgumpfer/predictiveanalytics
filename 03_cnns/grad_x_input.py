import matplotlib.pyplot as plt
import numpy as np
from signxai.methods.wrappers import calculate_relevancemap
from signxai.utils.utils import (load_image, aggregate_and_normalize_relevancemap_rgb)
from tensorflow.keras.applications.vgg16 import VGG16

# Load model
model = VGG16(weights='imagenet')

#  Remove last layer's softmax activation (we need the raw values!)
model.layers[-1].activation = None

# Load example image
img, x = load_image('../data/rooster.jpg')

# Calculate relevancemaps
R1 = calculate_relevancemap('gradient', np.array(x), model, neuron_selection=None)  # TODO: adjust neuron selection
R2 = calculate_relevancemap('gradient_x_input', np.array(x), model, neuron_selection=None)  # TODO: adjust neuron selection

# Visualize heatmaps
fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(18, 12))
axs[0].imshow(img)
axs[1].matshow(aggregate_and_normalize_relevancemap_rgb(R1), cmap='seismic', clim=(-1, 1))
axs[1].set_title('Gradient')
axs[2].matshow(aggregate_and_normalize_relevancemap_rgb(R2), cmap='seismic', clim=(-1, 1))
axs[2].set_title('Gradient x Input')

plt.tight_layout()

plt.show()
