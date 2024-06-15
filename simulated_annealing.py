import random
import time
import math
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

def generate_normal_neighborhood(permutation):
    a, b = random.sample(range(len(permutation)), 2)
    new_permutation = permutation[:]
    new_permutation[a], new_permutation[b] = new_permutation[b], new_permutation[a]
    return new_permutation

def temp_function(k, max_iterations):
    #return 1 - (k/max_iterations)
    return 0.5 * np.cos((k * np.pi) / (2 * max_iterations)) ** 1

def simulated_annealing(points, dist_matrix, alpha, iterations, show_plots=True):
    current_solution = random_solution(len(points))
    current_distance = route_length(current_solution, dist_matrix)
    best_solution = current_solution
    best_distance = current_distance
    
    distances = [current_distance]

    if show_plots:
        fig, ax = plt.subplots()
        plt.ion()

    start_time = time.time()

    for k in range(iterations):
        temp = temp_function(k, iterations)
        new_solution = generate_neighborhood_det(current_solution)
        new_distance = route_length(new_solution, dist_matrix)
        
        if new_distance < current_distance or math.exp((current_distance - new_distance) / temp) > np.random.rand():
            current_solution = new_solution
            current_distance = new_distance
            
            if new_distance < best_distance:
                best_solution = new_solution
                best_distance = new_distance
                
                if show_plots:
                    display_route(points, best_solution, ax)
                
        distances.append(current_distance)

    end_time = time.time()

    if show_plots:
        plt.ioff()
        plt.show()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")
    print(f"Final route length: {best_distance:.4f}")

    return best_solution, best_distance, distances

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Simulated Annealing for TSP')
    parser.add_argument('--cities', type=int, default=10, help='Number of cities')
    parser.add_argument('--iterations', type=int, default=10000, help='Number of iterations')
    parser.add_argument('--show_plots', action='store_true', help='Show plots during optimization')

    args = parser.parse_args()

    np.random.seed(42)
    points = np.random.uniform(-1, 1, (args.cities, 2))
    dist_matrix = distance_matrix(points)

    solution, distance, distances = simulated_annealing(points, dist_matrix, alpha=0.99, iterations=args.iterations, show_plots=args.show_plots)

    if not args.show_plots:
        plot_convergence_curve(distances, title='Simulated Annealing Convergence Curve')