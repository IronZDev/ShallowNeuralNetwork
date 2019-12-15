import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap, colorConverter
import numpy as np


def generate_plot(x):
    plt.scatter(x[0], x[1], c=x[2])
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    return plt.gcf()


def draw_decision_surface(network, x_min, x_max, y_min, y_max, num_of_x=200):
    num_of_x = int(num_of_x)
    # Create test set (surface)
    x0, x1 = np.meshgrid(np.linspace(x_min, x_max, num_of_x), np.linspace(y_min, y_max, num_of_x))
    # Calculate prediction for each point
    classification_plane = np.zeros((num_of_x, num_of_x))
    for i in range(num_of_x):
        for j in range(num_of_x):
            classification_plane[i, j] = np.around(
                network.predict(np.array([1, x0[i, j], x1[i, j]])))
    # Create color map
    cmap = ListedColormap([
        colorConverter.to_rgba('r', alpha=0.3),
        colorConverter.to_rgba('g', alpha=0.3)])
    # Plot the boundary
    plt.contourf(x0, x1, classification_plane, cmap=cmap)


# Draw plot on canvas
def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg
