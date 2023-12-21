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
img, x = load_image('../data/elephant.jpg')

# Calculate relevancemaps
R1 = calculate_relevancemap('gradient', np.array(x), model)
R2 = calculate_relevancemap('gradient_x_input', np.array(x), model)
R3 = calculate_relevancemap('gradient_x_sign', np.array(x), model)

R4 = calculate_relevancemap('gradient', np.array(x), model)
R5 = calculate_relevancemap('lrpz_epsilon_0_1_std_x', np.array(x), model)
R6 = calculate_relevancemap('lrpsign_epsilon_0_1_std_x', np.array(x), model)

# Visualize heatmaps
fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(18, 12))
axs[0][0].imshow(img)
axs[0][1].matshow(aggregate_and_normalize_relevancemap_rgb(R1), cmap='seismic', clim=(-1, 1))
axs[0][1].set_title('Gradient')
axs[0][2].matshow(aggregate_and_normalize_relevancemap_rgb(R2), cmap='seismic', clim=(-1, 1))
axs[0][2].set_title(r'Gradient $\times$ Input')
axs[0][3].matshow(aggregate_and_normalize_relevancemap_rgb(R3), cmap='seismic', clim=(-1, 1))
axs[0][3].set_title(r'Gradient $\times$ SIGN')

axs[1][0].imshow(img)
axs[1][1].matshow(aggregate_and_normalize_relevancemap_rgb(R4), cmap='seismic', clim=(-1, 1))
axs[1][1].set_title('Gradient')
axs[1][2].matshow(aggregate_and_normalize_relevancemap_rgb(R5), cmap='seismic', clim=(-1, 1))
axs[1][2].set_title(r'LRP-$\epsilon$ (z-Rule)')
axs[1][3].matshow(aggregate_and_normalize_relevancemap_rgb(R6), cmap='seismic', clim=(-1, 1))
axs[1][3].set_title(r'LRP-$\epsilon$ (SIGN)')

plt.tight_layout()

plt.show()
