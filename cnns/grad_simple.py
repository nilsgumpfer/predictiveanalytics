import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def plot_function_surface(x1, x2, Z, C, t, cmap='gist_earth'):
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(projection='3d')
    vmax = np.max(np.abs(np.ravel(C)))
    vmin = -vmax
    norm = Normalize(vmin, vmax)
    scamap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    ax.plot_surface(x1, x2, Z, facecolors=scamap.to_rgba(C), cmap=cmap)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('z')
    ax.set_title(t)
    fig.colorbar(scamap, shrink=0.5, aspect=10)
    plt.tight_layout()


def f(x1, x2):
    return x2 * np.sin(x1 ** 2)
    # return (x1**2) * x2


def df_x1(x1, x2):
    return 2 * x1 * x2 * np.cos(x1 ** 2)
    # return 2 * x1 * x2


def df_x2(x1, x2):
    return np.sin(x1 ** 2)
    # return x1 ** 2


def main():
    x1 = np.linspace(-2, 2, 50)
    x2 = np.linspace(-5, 5, 50)
    x1, x2 = np.meshgrid(x1, x2)

    # Function plot
    plot_function_surface(x1=x1, x2=x2, Z=f(x1, x2), C=f(x1, x2), t='Z=f(x1, x2), C=f(x1, x2)')

    # Function plots with x1 and x2 gradients
    # plot_function_surface(x1=x1, x2=x2, Z=f(x1, x2), C=df_x1(x1, x2), t='Z=f(x1, x2), C=df_x1(x1, x2)')
    # plot_function_surface(x1=x1, x2=x2, Z=f(x1, x2), C=df_x2(x1, x2), t='Z=f(x1, x2), C=df_x2(x1, x2)')

    # Function plots with x1 gradient * input and x2 gradient * input
    # plot_function_surface(x1=x1, x2=x2, Z=f(x1, x2), C=df_x1(x1, x2) * x1, t='Z=f(x1, x2), C=df_x1(x1, x2) * x1')
    # plot_function_surface(x1=x1, x2=x2, Z=f(x1, x2), C=df_x2(x1, x2) * x2, t='Z=f(x1, x2), C=df_x2(x1, x2) * x2')

    # Gradient plots
    # plot_function_surface(x1=x1, x2=x2, Z=df_x1(x1, x2), C=df_x1(x1, x2), t='Z=df_x1(x1, x2), C=df_x1(x1, x2)')
    # plot_function_surface(x1=x1, x2=x2, Z=df_x2(x1, x2), C=df_x2(x1, x2), t='Z=df_x2(x1, x2), C=df_x2(x1, x2)')

    # Gradient * input plots
    # plot_function_surface(x1=x1, x2=x2, Z=df_x1(x1, x2) * x1, C=df_x1(x1, x2) * x1, t='Z=df_x1(x1, x2) * x1, C=df_x1(x1, x2) * x1')
    # plot_function_surface(x1=x1, x2=x2, Z=df_x2(x1, x2) * x2, C=df_x2(x1, x2) * x2, t='Z=df_x2(x1, x2) * x2, C=df_x2(x1, x2) * x2')

    plt.show()

main()
