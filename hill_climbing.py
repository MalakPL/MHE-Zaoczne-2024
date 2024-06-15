import random
import time
import numpy as np
import matplotlib.pyplot as plt
from utils import distance_matrix, route_length, display_route, plot_convergence_curve

def random_solution(n_cities):
    permutation = list(range(n_cities))
    random.shuffle(permutation)
    return permutation

def generate_neighborhood(permutation):
    a, b = random.sample(range(len(permutation)), 2)
    new_permutation = permutation[:]
    new_permutation[a], new_permutation[b] = new_permutation[b], new_permutation[a]
    return new_permutation

def generate_neighborhood_det(permutation):
    neighborhood = []
    for i in range(len(permutation)):
        for j in range(i + 1, len(permutation)):
            new_permutation = permutation[:]
            new_permutation[i], new_permutation[j] = new_permutation[j], new_permutation[i]
            neighborhood.append(new_permutation)

    return  min(neighborhood, key=lambda perm: route_length(perm, dist_matrix))

def hill_climbing(points, dist_matrix, iterations=100, show_plots=True):
    current_solution = random_solution(len(points))
    current_length = route_length(current_solution, dist_matrix)
    distances = []

    if show_plots:
        fig, ax = plt.subplots()
        plt.ion()

    start_time = time.time()

    for _ in range(iterations):
        new_solution = generate_neighborhood_det(current_solution)
        new_length = route_length(new_solution, dist_matrix)
        
        if new_length < current_length:
            current_solution = new_solution
            current_length = new_length
        
        distances.append(current_length)
        
        if show_plots:
            display_route(points, current_solution, ax)

    end_time = time.time()

    if show_plots:
        plt.ioff()
        plt.show()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")
    print(f"Final route length: {current_length:.4f}")

    return current_solution, current_length, distances

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Hill Climbing for TSP')
    parser.add_argument('--cities', type=int, default=10, help='Number of cities')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations')
    parser.add_argument('--show_plots', action='store_true', help='Show plots during optimization')

    args = parser.parse_args()

    np.random.seed(42)
    points = np.random.uniform(-1, 1, (args.cities, 2))
    dist_matrix = distance_matrix(points)

    solution, distance, distances = hill_climbing(points, dist_matrix, args.iterations, args.show_plots)

    if not args.show_plots:
        plot_convergence_curve(distances, title='Hill Climbing Convergence Curve')