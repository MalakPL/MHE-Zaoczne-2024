import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

def distance_matrix(points):
    return squareform(pdist(points, 'euclidean'))

def route_length(permutation, dist_matrix):
    length = 0
    num_points = len(permutation)
    for i in range(num_points - 1):
        length += dist_matrix[permutation[i], permutation[(i + 1)]]
    length += dist_matrix[permutation[-1], permutation[0]]
    return length

def display_route(points, route, ax):
    ax.clear()
    ax.scatter(points[:, 0], points[:, 1], color='blue')
    ax.plot(points[route, 0], points[route, 1], color='red', marker='o')
    for i, txt in enumerate(range(len(points))):
        ax.annotate(txt, (points[i, 0], points[i, 1]))
    ax.set_title('Traveling Salesman Problem')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    plt.draw()
    plt.pause(0.001)

def simple_moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_convergence_curve(distances, title='Convergence Curve'):
    smoothed_distances = simple_moving_average(distances, window_size=100)
    plt.plot(smoothed_distances, label='SMA (window=100)', alpha=0.8)
    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.title(title)
    plt.legend()
    plt.show()
