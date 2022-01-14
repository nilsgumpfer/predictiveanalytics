# visualize feature maps output from each block in the vgg model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
import matplotlib.pyplot as plt
from numpy import expand_dims
import numpy as np


model = VGG16()
layer_indices = np.arange(start=1, stop=len(model.layers))
outputs = [model.layers[i].output for i in layer_indices]
model = Model(inputs=model.inputs, outputs=outputs)

img = load_img('../data/giraffe.jpg', target_size=(224, 224))
img = img_to_array(img)
img = expand_dims(img, axis=0)
img = preprocess_input(img)

all_layers_feature_maps = model.predict(img)

for idx, feature_maps in zip(layer_indices, all_layers_feature_maps):
    lshape = np.shape(feature_maps)
    nfeaturemaps = lshape[-1]
    dims = len(lshape)
    square = int(nfeaturemaps ** 0.5)

    # retrieve weights from the second hidden layer
    w = model.layers[idx].get_weights()
    if len(w) == 2:
        filters, biases = w
        print(len(filters))


        # if dims == 4:
        #     # Visualize filters
        #     fig, axs = plt.subplots(nrows=square, ncols=square, figsize=(15, 15))
        #     fig.patch.set_facecolor('black')
        #
        #     f = 0
        #     for r in range(square):
        #         for c in range(square):
        #             axs[r][c].imshow(filters[0, :, :, f], cmap='gray')
        #             axs[r][c].axis('off')
        #             f += 1
        #
        #     plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        #     plt.savefig('../data/plots/filters_{}.jpg'.format(idx))
        #     plt.close()
        #     #
        #     # # Visualize feature maps
        #     # fig, axs = plt.subplots(nrows=square, ncols=square, figsize=(15, 15))
        #     # fig.patch.set_facecolor('black')
        #     #
        #     # f = 0
        #     # for r in range(square):
        #     #     for c in range(square):
        #     #         axs[r][c].imshow(feature_maps[0, :, :, f], cmap='gray')
        #     #         axs[r][c].axis('off')
        #     #         f += 1
        #     #
        #     # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        #     # plt.savefig('../data/plots/feature_maps_{}.jpg'.format(idx))
        #     # plt.close()
