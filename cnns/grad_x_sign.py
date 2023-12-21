import numpy as np
import matplotlib.pyplot as plt
# from keras.src.applications import VGG16
from tensorflow.keras.applications.vgg16 import VGG16
from signxai.methods.wrappers import calculate_relevancemap
from signxai.utils.utils import (load_image, aggregate_and_normalize_relevancemap_rgb, download_image,
                                 calculate_explanation_innvestigate)

# Load model
model = VGG16(weights='imagenet')

#  Remove last layer's softmax activation (we need the raw values!)
model.layers[-1].activation = None

# Load example image
img, x = load_image(path)

# Calculate relevancemaps
R1 = calculate_relevancemap('lrpz_epsilon_0_1_std_x', np.array(x), model)
R2 = calculate_relevancemap('lrpsign_epsilon_0_1_std_x', np.array(x), model)

# Visualize heatmaps
fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(18, 12))
axs[0][0].imshow(img)
axs[1][0].imshow(img)
axs[0][1].matshow(aggregate_and_normalize_relevancemap_rgb(R1), cmap='seismic', clim=(-1, 1))
axs[0][2].matshow(aggregate_and_normalize_relevancemap_rgb(R2), cmap='seismic', clim=(-1, 1))
axs[1][1].matshow(aggregate_and_normalize_relevancemap_rgb(R3), cmap='seismic', clim=(-1, 1))
axs[1][2].matshow(aggregate_and_normalize_relevancemap_rgb(R4), cmap='seismic', clim=(-1, 1))

plt.show()
