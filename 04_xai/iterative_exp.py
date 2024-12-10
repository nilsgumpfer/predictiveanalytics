import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageEnhance
from signxai.methods.signed import calculate_sign_mu
from signxai.methods.wrappers import calculate_relevancemap
from signxai.utils.utils import (load_image, aggregate_and_normalize_relevancemap_rgb)
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.python.ops.numpy_ops.np_random import random
from PIL import Image


def generate_animation_from_plots(target_path, paths, cleanup=False):
    # Load images into a list
    frames = [Image.open(img) for img in paths]

    # Save the images as an animated WEBP
    frames[0].save(
        target_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        format="WEBP"
    )

    if cleanup:
        for x in paths:
            os.remove(x)

def reverse_preprocess_image(x, dtype=int):
    # Undo zero-centering based on ImageNet mean RGB values
    mean = [103.939, 116.779, 123.68]
    x[..., 0] += mean[0]
    x[..., 1] += mean[1]
    x[..., 2] += mean[2]

    # 'BGR'->'RGB'
    x = x[..., ::-1]

    return np.array(x, dtype=dtype)


def preprocess_image(x):
    # 'RGB'->'BGR'
    x = x[..., ::-1]

    # Zero-centering based on ImageNet mean RGB values
    mean = [103.939, 116.779, 123.68]
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]

    return x


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


def main(img_name, amplify=True):
    # Load model
    model = VGG16(weights='imagenet')
    model_softmax = VGG16(weights='imagenet')

    #  Remove last layer's softmax activation (we need the raw values!)
    model.layers[-1].activation = None

    # Load example image
    img, x = get_image('../data/{}'.format(img_name))

    # Predict
    neuron_selection = int(np.argmax(model_softmax(np.array([x]))))

    # Gradient placeholder
    G = np.zeros_like(x)

    paths = []

    for i in range(150):
        print(i)

        # Apply grad to x
        if i > 0:
            if amplify:
                x = x + 16 * (G / np.max(np.abs(np.ravel(G))))
            else:
                x = x - 16 * (G / np.max(np.abs(np.ravel(G))))

            x = clip_x(x)

        # Calculate gradient
        G = calculate_relevancemap('gradient', np.array(x), model, neuron_selection=neuron_selection)
        pred = model_softmax(np.array([x]))[-1][neuron_selection]

        # Visualize result
        fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(18, 7))
        axs[0].set_title('Image')
        axs[0].imshow(reverse_preprocess_image(np.array(x)))
        axs[1].set_title('Pos/Neg')
        axs[1].imshow(aggregate_and_normalize_relevancemap_rgb(x), cmap='seismic', clim=(-1, 1))
        axs[2].set_title('Gradient')
        axs[2].imshow(aggregate_and_normalize_relevancemap_rgb(G), cmap='seismic', clim=(-1, 1))
        plt.suptitle('Iteration {}, Prediction for idx={}: {:.2f}'.format(i, neuron_selection, float(pred)))

        plt.tight_layout()
        plot_path = '../data/plots/iter_{}_{}.jpg'.format(amplify, i)
        plt.savefig(plot_path)
        paths.append(plot_path)

    generate_animation_from_plots('../data/plots/iter_{}_{}.webp'.format(img_name[:-4], amplify), paths, cleanup=True)

def generate(img_name, neuron_selection, randm=False):
    # Load model
    model = VGG16(weights='imagenet')
    model_softmax = VGG16(weights='imagenet')

    #  Remove last layer's softmax activation (we need the raw values!)
    model.layers[-1].activation = None

    # Base image
    if randm:
        # x = np.random.normal(loc=0, scale=64, size=(224, 224, 3))
        x = np.zeros((224, 224, 3))
        c = np.random.uniform(low=-8, high=8, size=(224, 224))
        x[..., 0] = c
        x[..., 1] = c
        x[..., 2] = c
    else:
        x = np.zeros((224, 224, 3))

    # Gradient placeholder
    G = np.zeros_like(x)

    paths = []

    for i in range(150):
        print(i)

        # Apply grad to x
        if i > 0:
            x = x + 16 * (G / np.max(np.abs(np.ravel(G))))
            x = clip_x(x)

        # Calculate gradient
        G = calculate_relevancemap('gradient', np.array(x), model, neuron_selection=neuron_selection)
        pred = model_softmax(np.array([x]))[-1][neuron_selection]

        # Visualize result
        fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(18, 7))
        axs[0].set_title('Image')
        axs[0].imshow(reverse_preprocess_image(np.array(x)))
        axs[1].set_title('Pos/Neg')
        axs[1].imshow(aggregate_and_normalize_relevancemap_rgb(x), cmap='seismic', clim=(-1, 1))
        axs[2].set_title('Gradient')
        axs[2].imshow(aggregate_and_normalize_relevancemap_rgb(G), cmap='seismic', clim=(-1, 1))
        plt.suptitle('Iteration {}, Prediction for idx={}: {:.2f}'.format(i, neuron_selection, float(pred)))

        plt.tight_layout()
        plot_path = '../data/plots/gen_{}_{}_{}.jpg'.format({True: 'rand', False: 'zeros'}[randm], neuron_selection, i)
        plt.savefig(plot_path)
        paths.append(plot_path)

    generate_animation_from_plots('../data/plots/gen_{}_{}_idx{}.webp'.format(img_name, {True: 'rand', False: 'zeros'}[randm], neuron_selection), paths, cleanup=True)


def clip_x(x):
    tmp = reverse_preprocess_image(np.array(x), dtype=float)
    return preprocess_image(np.clip(tmp, 0, 255))


def adjust(img_name, trgt_idx, n=100):
    # Load model
    model = VGG16(weights='imagenet')
    model_softmax = VGG16(weights='imagenet')

    #  Remove last layer's softmax activation (we need the raw values!)
    model.layers[-1].activation = None

    # Load example image
    img, x = get_image('../data/{}'.format(img_name))

    # Gradient placeholder
    G = np.zeros_like(x)

    paths = []

    for i in range(n):
        print(i)

        # Apply grad to x
        if i > 0:
            x = x + 16 * (G / np.max(np.abs(np.ravel(G))))
            x = clip_x(x)

        # Calculate gradient
        G = calculate_relevancemap('gradient', np.array(x), model, neuron_selection=trgt_idx)
        pred = model_softmax(np.array([x]))[-1][trgt_idx]

        # Visualize result
        fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(18, 7))
        axs[0].set_title('Image')
        axs[0].imshow(reverse_preprocess_image(np.array(x)))
        axs[1].set_title('Pos/Neg')
        axs[1].imshow(aggregate_and_normalize_relevancemap_rgb(x), cmap='seismic', clim=(-1, 1))
        axs[2].set_title('Gradient')
        axs[2].imshow(aggregate_and_normalize_relevancemap_rgb(G), cmap='seismic', clim=(-1, 1))
        plt.suptitle('Iteration {}, Prediction for idx={}: {:.2f}'.format(i, trgt_idx, float(pred)))

        plt.tight_layout()
        plot_path = '../data/plots/adjust_{}_{}.jpg'.format(trgt_idx, i)
        plt.savefig(plot_path)
        paths.append(plot_path)

    generate_animation_from_plots('../data/plots/adjust_{}_{}.webp'.format(img_name[:-4], trgt_idx), paths, cleanup=True)


def explain(img_name, trgt_idx, n=10, method='gradient'):
    # Load model
    model = VGG16(weights='imagenet')

    #  Remove last layer's softmax activation (we need the raw values!)
    model.layers[-1].activation = None

    # Load example image
    img, x = get_image('../data/{}'.format(img_name))

    # Gradient placeholder
    G = np.zeros_like(x)

    explanations = np.zeros((224, 224, 3, n))

    for i in range(n):
        print(i)

        # Apply grad to x
        if i > 0:
            x = x + 16 * (G / np.max(np.abs(np.ravel(G))))
            x = clip_x(x)

        # Calculate gradient
        if method == 'gradient':
            G = calculate_relevancemap('gradient', np.array(x), model, neuron_selection=trgt_idx)
        else:
            G = calculate_relevancemap(method, np.array(x), model, neuron_selection=trgt_idx) / np.array(x)

        # SIGN-adjustment
        explanations[..., i] = G * calculate_sign_mu(x)


    # Visualize result
    fig, axs = plt.subplots(ncols=4, nrows=1, figsize=(24, 6))
    axs[0].set_title('Image 0')
    axs[0].imshow(img)
    axs[1].set_title('Gradient 0')
    axs[1].imshow(aggregate_and_normalize_relevancemap_rgb(explanations[..., 0]), cmap='seismic', clim=(-1, 1))
    axs[2].set_title('Image gen')
    axs[2].imshow(reverse_preprocess_image(np.array(x)))
    axs[3].set_title('Gradient mean')
    axs[3].imshow(aggregate_and_normalize_relevancemap_rgb(np.mean(explanations, axis=2)), cmap='seismic', clim=(-1, 1))

    plt.tight_layout()
    plot_path = '../data/plots/explain_{}_{}_{}.jpg'.format(img_name[:-4], trgt_idx, method)
    plt.savefig(plot_path)
    plt.close()


if __name__ == '__main__':
    # adjust('hen3.jpg', 7) #--> disputation
    # main('tigershark3.png', False)
    # main('tigershark3.png', True)
    # adjust('giraffe.jpg', 130)
    # generate('flamingo', 130, randm=True)
    # main('cobra.png', True)

    # adjust('impalas.png', 352)

    # adjust('forest.png', 483) # --> disputation?
    # adjust('eltz.jpg', 483)

    # for m in ['gradient', 'lrpz_epsilon_0_1_std_x', 'lrpz_epsilon_0_25_std_x']:
    #     explain('impalas.png', 352, n=10, method=m)
    #     explain('rooster.jpg', 7, n=10, method=m)
    #     explain('hen3.jpg', 7, n=10, method=m)
    #     explain('castlebicycle.jpg', 483, n=10, method=m)
    #     explain('castlebicycle.jpg', 671, n=10, method=m)
    #     explain('elephant.jpg', 386, n=10, method=m)
    #     explain('cobra.png', 63, n=10, method=m)
    #     explain('eltz2.png', 483, n=4, method=m)
    #     explain('bodiamcastle.jpg', 483, n=10, method=m)

    adjust('storch2.jpg', 130, n=200)

    # TODO: mean gradient over adjustment iterations

