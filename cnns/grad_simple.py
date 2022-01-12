import math
import os
from math import tanh

import imageio as imageio
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn.metrics import mean_squared_error
from matplotlib import cm


def plot_function_surface(X, Y, Z, C, t, cmap='gist_earth'):
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(projection='3d')
    scamap = plt.cm.ScalarMappable(cmap=cmap)
    ax.plot_surface(X, Y, Z, facecolors=scamap.to_rgba(C), cmap=cmap)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(t)
    fig.colorbar(scamap, shrink=0.5, aspect=10)
    plt.tight_layout()


def f(x, y):
    # return y * np.sin(x ** 2)
    return (x**2) * y


def df_x(x, y):
    # return 2 * x * y * np.cos(x ** 2)
    return 2 * x * y


def df_y(x, y):
    # return np.sin(x ** 2)
    return x ** 2


def main():
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-5, 5, 50)
    x, y = np.meshgrid(x, y)

    plot_function_surface(X=x, Y=y, Z=f(x, y), C=f(x, y), t='Z=f(x, y), C=f(x, y)')
    plot_function_surface(X=x, Y=y, Z=f(x, y), C=df_x(x, y), t='Z=f(x, y), C=df_x(x, y)')
    plot_function_surface(X=x, Y=y, Z=f(x, y), C=df_y(x, y), t='Z=f(x, y), C=df_y(x, y)')

    # plot_function_surface(X=x, Y=y, Z=df_x(x, y), C=f(x, y), t='Z=df_x(x, y), C=f(x, y)')
    # plot_function_surface(X=x, Y=y, Z=df_x(x, y), C=df_x(x, y), t='Z=df_x(x, y), C=df_x(x, y)')
    # plot_function_surface(X=x, Y=y, Z=df_x(x, y), C=df_y(x, y), t='Z=df_x(x, y), C=df_y(x, y)')

    # plot_function_surface(X=x, Y=y, Z=df_y(x, y), C=f(x, y), t='Z=df_y(x, y), C=f(x, y)')
    # plot_function_surface(X=x, Y=y, Z=df_y(x, y), C=df_x(x, y), t='Z=df_y(x, y), C=df_x(x, y)')
    # plot_function_surface(X=x, Y=y, Z=df_y(x, y), C=df_y(x, y), t='Z=df_y(x, y), C=df_y(x, y)')

    plt.show()

main()
