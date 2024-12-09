import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageEnhance
from signxai.methods.signed import calculate_sign_mu
from signxai.methods.wrappers import calculate_relevancemap
from signxai.utils.utils import (load_image, aggregate_and_normalize_relevancemap_rgb)
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image

def reverse_preprocess_image(x):
    # Undo zero-centering based on ImageNet mean RGB values
    mean = [103.939, 116.779, 123.68]
    x[..., 0] += mean[0]
    x[..., 1] += mean[1]
    x[..., 2] += mean[2]

    # 'BGR'->'RGB'
    x = x[..., ::-1]

    return np.array(x, dtype=int)


def get_image(img_path, brightness=1.0, contrast=1.0, expand_dims=False):
    # Load image
    img = image.load_img(img_path, target_size=(224, 224))

    # Adjust contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast)

    # Array conversion
    x = image.img_to_array(img)

    # Adjust brightness
    x = x * brightness
    x = np.clip(x, a_min=0, a_max=255)
    img = np.array(x, dtype=int)

    if expand_dims:
        x = np.expand_dims(x, axis=0)

    # 'RGB'->'BGR'
    x = x[..., ::-1]

    # Zero-centering based on ImageNet mean RGB values
    mean = [103.939, 116.779, 123.68]
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]

    return img, x


# def gradient_by_input(g, x):
#     res = g * calculate_sign_mu(x)
#     res[]
# TODO: sensitivity != relevance -> dark areas can have high sensitivity (pos+neg) to induce patterns
# TODO: mu could be rooted in separation value of convolutional filters (dark to bright)



def main():
    # Load model
    model = VGG16(weights='imagenet')

    #  Remove last layer's softmax activation (we need the raw values!)
    model.layers[-1].activation = None

    # Load example image
    img, x = get_image('../data/cobra.png')
    # img, x = get_image('../data/zebra.jpeg')
    # img, x = get_image('../data/rooster.jpg', contrast=0.5, brightness=1.2)
    # img, x = get_image('../data/Screenshot from 2024-12-06 15-31-11.png', contrast=0.9)
    # img, x = get_image('../data/tigershark.jpg', contrast=0.9)
    # img, x = get_image('../data/zebra.jpeg', contrast=0.9)

    # Calculate relevancemaps
    R0 = calculate_relevancemap('gradient_x_input', np.array(x), model, neuron_selection=None)
    R1 = calculate_relevancemap('gradient', np.array(x), model, neuron_selection=None)
    R2 = calculate_relevancemap('gradient_x_sign', np.array(x), model, neuron_selection=None)

    R1_n = R1 / np.max(np.abs(np.ravel(R1)))
    x_grad = x + 512 * R1_n
    img_grad = reverse_preprocess_image(np.array(x_grad))

    R0 = aggregate_and_normalize_relevancemap_rgb(R0)
    R1 = aggregate_and_normalize_relevancemap_rgb(R1)
    R2 = aggregate_and_normalize_relevancemap_rgb(R2)

    # Visualize heatmaps
    fig, axs = plt.subplots(ncols=5, nrows=2, figsize=(30, 12))
    axs[0][0].imshow(img)
    axs[1][0].imshow(img_grad)

    # axs[0][1].set_title('LRP-z')
    axs[0][1].set_title('Gradient x Input')
    axs[0][1].matshow(R0, cmap='seismic', clim=(-1, 1))
    axs[1][1].imshow(img_grad)

    # axs[0][2].set_title('LRP-z / x')
    axs[0][2].set_title('Gradient')
    axs[0][2].matshow(R1, cmap='seismic', clim=(-1, 1))
    axs[1][2].imshow(img_grad)

    # axs[0][3].set_title('LRP-SIGN')
    axs[0][3].set_title('Gradient x SIGN')
    axs[0][3].matshow(R2, cmap='seismic', clim=(-1, 1))
    axs[1][3].imshow(img_grad)

    axs[0][4].set_title('Pos/Neg')
    axs[0][4].matshow(aggregate_and_normalize_relevancemap_rgb(x), cmap='seismic', clim=(-1, 1))
    axs[1][4].matshow(aggregate_and_normalize_relevancemap_rgb(x_grad), cmap='seismic', clim=(-1, 1))

    plt.tight_layout()
    plt.savefig('example.pdf', dpi=500)


def channel_wise():
    # Load model
    model = VGG16(weights='imagenet')

    #  Remove last layer's softmax activation (we need the raw values!)
    model.layers[-1].activation = None

    # Load example image
    # img, x = get_image('../data/rooster.jpg')
    # img, x = get_image('../data/zebra.jpeg')
    img, x = get_image('../data/cobra.png')

    # Visualize heatmaps
    fig, axs = plt.subplots(ncols=6, nrows=3, figsize=(30, 15))

    # Channel-wise iteration
    for ch in [0, 1, 2]:
        cmap = {0: 'Blues', 1: 'Greens', 2: 'Reds'}[ch]

        R0 = calculate_relevancemap('gradient_x_input', np.array(x), model, neuron_selection=None)[..., ch]
        R1 = calculate_relevancemap('gradient', np.array(x), model, neuron_selection=None)[..., ch]
        R2 = calculate_relevancemap('gradient_x_sign', np.array(x), model, neuron_selection=None)[..., ch]

        R0 = R0 / np.max(np.abs(np.ravel(R0)))
        R1 = R1 / np.max(np.abs(np.ravel(R1)))
        R2 = R2 / np.max(np.abs(np.ravel(R2)))
        x_ch = x[..., ch] / np.max(np.abs(np.ravel(x[..., ch])))

        axs[ch][0].imshow(img)
        axs[ch][1].matshow(img[..., ch], cmap=cmap, clim=(0, 255))

        axs[ch][2].set_title('Gradient x Input')
        axs[ch][2].matshow(R0, cmap='seismic', clim=(-1, 1))

        axs[ch][3].set_title('Gradient')
        axs[ch][3].matshow(R1, cmap='seismic', clim=(-1, 1))

        axs[ch][4].set_title('Gradient x SIGN')
        axs[ch][4].matshow(R2, cmap='seismic', clim=(-1, 1))

        axs[ch][5].set_title('Pos/Neg')
        axs[ch][5].matshow(x_ch, cmap='seismic', clim=(-1, 1))

    plt.tight_layout()
    plt.savefig('channel_wise.pdf'.format(ch), dpi=500)

if __name__ == '__main__':
    main()
    # channel_wise()