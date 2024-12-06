import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageEnhance
from signxai.methods.wrappers import calculate_relevancemap
from signxai.utils.utils import (load_image, aggregate_and_normalize_relevancemap_rgb)
from tensorflow.keras.applications.vgg16 import VGG16

def reverse_preprocess_image(x):
    # Undo zero-centering based on ImageNet mean RGB values
    mean = [103.939, 116.779, 123.68]
    x[..., 0] += mean[0]
    x[..., 1] += mean[1]
    x[..., 2] += mean[2]

    # 'BGR'->'RGB'
    x = x[..., ::-1]

    return np.array(x, dtype=int)

def main():
    # Load model
    model = VGG16(weights='imagenet')

    #  Remove last layer's softmax activation (we need the raw values!)
    model.layers[-1].activation = None

    # Load example image
    # img, x = load_image('../data/Screenshot from 2024-12-06 15-31-11.png')
    img, x = load_image('../data/tigershark.jpg')

    # Calculate relevancemaps
    # R0 = calculate_relevancemap('lrpz_epsilon_0_1_std_x', np.array(x), model, neuron_selection=None)
    # R1 = R0 / np.array(x)
    # R2 = calculate_relevancemap('lrpsign_epsilon_0_1_std_x', np.array(x), model, neuron_selection=None)

    R0 = calculate_relevancemap('gradient_x_input', np.array(x), model, neuron_selection=None)
    R1 = calculate_relevancemap('gradient', np.array(x), model, neuron_selection=None)
    R2 = calculate_relevancemap('gradient_x_sign', np.array(x), model, neuron_selection=None)

    R1_n = R1 / np.max(np.abs(np.ravel(R1)))
    x_grad = x + 255 * R1_n
    x_grad = reverse_preprocess_image(x_grad)

    R0 = aggregate_and_normalize_relevancemap_rgb(R0)
    R1 = aggregate_and_normalize_relevancemap_rgb(R1)
    R2 = aggregate_and_normalize_relevancemap_rgb(R2)

    img2 = ImageEnhance.Contrast(img).enhance(0.1)
    img = np.array(img)

    R0_img = np.array(img2)
    R0_img[R0 > 0.05] = img[R0 > 0.05]
    # R0[R0 > 0.05] = R0[R0 > 0.05] * 3

    R1_img = np.array(img2)
    R1_img[R1 > 0.05] = img[R1 > 0.05]
    # R1[R1 > 0.05] = R1[R1 > 0.05] * 3

    R2_img = np.array(img2)
    R2_img[R2 > 0.05] = img[R2 > 0.05]
    # R2[R2 > 0.05] = R2[R2 > 0.05] * 3

    # Visualize heatmaps
    fig, axs = plt.subplots(ncols=5, nrows=2, figsize=(25, 20))
    axs[0][0].imshow(img)
    axs[1][0].imshow(x_grad)

    # axs[0][1].set_title('LRP-z')
    axs[0][1].set_title('Gradient x Input')
    axs[0][1].matshow(R0, cmap='seismic', clim=(-1, 1))
    axs[1][1].imshow(R0_img)

    # axs[0][2].set_title('LRP-z / x')
    axs[0][2].set_title('Gradient')
    axs[0][2].matshow(R1, cmap='seismic', clim=(-1, 1))
    axs[1][2].imshow(R1_img)

    # axs[0][3].set_title('LRP-SIGN')
    axs[0][3].set_title('Gradient x SIGN')
    axs[0][3].matshow(R2, cmap='seismic', clim=(-1, 1))
    axs[1][3].imshow(R2_img)

    axs[0][4].set_title('Pos/Neg')
    axs[0][4].matshow(aggregate_and_normalize_relevancemap_rgb(x), cmap='seismic', clim=(-1, 1))
    axs[1][4].matshow(aggregate_and_normalize_relevancemap_rgb(x), cmap='seismic', clim=(-1, 1))


    plt.tight_layout()

    # plt.show()
    plt.savefig('example.pdf', dpi=500)


if __name__ == '__main__':
    main()