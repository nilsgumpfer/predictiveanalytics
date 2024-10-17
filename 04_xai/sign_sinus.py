import math
import os

import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.sin(x)


def gradient(x):
    return np.cos(x)


def gradient_x_input(x):
    return gradient(x) * x

def gradient_x_sign(x):
    return gradient(x) * sign(x)


def sign(x):
    return x / np.abs(x)


def run_example_1d_simple(pirange=5):
    path = 'plots/1Dexample'
    os.makedirs(path, exist_ok=True)

    xlow, xhigh = -pirange*math.pi, pirange*math.pi
    x = np.linspace(xlow, xhigh, 3001)

    xticks = np.arange(xlow, xhigh+0.01, step=math.pi)
    xticklabels = [r'${}\pi$'.format(int(round(x, 2) / round(math.pi, 2))) for x in xticks]

    fig, axs = plt.subplots(nrows=4, figsize=(8, 5))
    axs[0].plot(x, f(x), c='black')
    axs[0].set_title('Function')
    axs[0].set_ylabel(r'$f(x)$', rotation=0, horizontalalignment="right", verticalalignment="center")
    axs[1].plot(x, gradient(x), c='blue')
    axs[1].set_title('Gradient')
    axs[1].set_ylabel(r'$\frac{\sigma f}{\sigma x}(x)$', rotation=0, horizontalalignment="right", verticalalignment="center")
    axs[2].plot(x, gradient_x_input(x), c='blue')
    axs[2].set_title(r'Gradient $\times$ Input')
    axs[2].set_ylabel(r'$\frac{\sigma f}{\sigma x}(x) \cdot x$', rotation=0, horizontalalignment="right", verticalalignment="center")
    axs[3].set_title(r'Gradient $\times$ SIGN')
    axs[3].plot(x, gradient_x_sign(x), c='blue')
    axs[3].set_ylabel(r'$\frac{\sigma f}{\sigma x}(x) \cdot x$', rotation=0, horizontalalignment="right", verticalalignment="center")

    for ax in axs:
        ax.set_xlim((xticks[0], xticks[-1]))
        ax.set_xlabel(r'$x$')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

    plt.tight_layout()
    # plt.savefig('{}/example_1d_simple_functions'.format(path))
    plt.show()
    plt.close()


run_example_1d_simple()