import itertools
import time
import numpy as np
import matplotlib.pyplot as plt
from utils import distance_matrix, route_length, display_route, plot_convergence_curve

def brute_force(points, dist_matrix, show_plots):
    n_cities = len(points)
    all_permutations = itertools.permutations(range(n_cities))
    min_length = float('inf')
    best_route = None
    distances = []

    start_time = time.time()

    if show_plots:
        plt.ion()
        fig, ax = plt.subplots()

    for perm in all_permutations:
        current_length = route_length(perm, dist_matrix)
        if current_length < min_length:
            min_length = current_length
            best_route = perm

            if show_plots:
                ax.clear()
                display_route(points, best_route, ax)
        
        distances.append(min_length)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time:.4f} seconds")
    print(f"Final route length: {min_length:.4f}")

    if show_plots:
        plt.ioff()

    return best_route, min_length, distances

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Brute Force for TSP')
    parser.add_argument('--cities', type=int, default=10, help='Number of cities')
    parser.add_argument('--show_plots', action='store_true', help='Show plots during optimization')

    args = parser.parse_args()

    np.random.seed(42)
    points = np.random.uniform(-1, 1, (args.cities, 2))
    dist_matrix = distance_matrix(points)

    solution, distance, distances = brute_force(points, dist_matrix, show_plots=args.show_plots)

    plot_convergence_curve(distances, title='Brute Force Convergence Curve')