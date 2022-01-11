import math
import os
from math import tanh

import imageio as imageio
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn.metrics import mean_squared_error
from matplotlib import cm


def plot_function_surface(X, Y, Z):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection='3d')
    facecolors = cm.viridis(Z/np.amax(Z))
    # ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.plot_surface(X, Y, Z, facecolors=facecolors)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def f(x, y):
    return y * np.sin(x ** 2)


def df_x(x, y):
    return 2 * x * y * np.cos(x ** 2)


def df_y(x, y):
    return np.sin(x ** 2)


def main():
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-5, 5, 100)
    x, y = np.meshgrid(x, y)

    plot_function_surface(x, y, f(x, y))
    # plot_function_surface(x, y, df_x(x, y))
    # plot_function_surface(x, y, df_y(x, y))


main()
