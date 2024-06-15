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

def tabu_search(points, dist_matrix, tabu_size, iterations, show_plots=True):
    current_solution = random_solution(len(points))
    best_solution = current_solution
    best_distance = route_length(best_solution, dist_matrix)
    
    tabu_list = []
    distances = [best_distance]

    if show_plots:
        fig, ax = plt.subplots()
        plt.ion()

    start_time = time.time()

    for _ in range(iterations):
        neighborhood = [generate_neighborhood(current_solution) for _ in range(100)]
        neighborhood = [perm for perm in neighborhood if perm not in tabu_list]
        
        if not neighborhood:
            break
        
        neighborhood_distances = [route_length(perm, dist_matrix) for perm in neighborhood]
        best_neighbor_index = neighborhood_distances.index(min(neighborhood_distances))
        best_neighbor = neighborhood[best_neighbor_index]
        
        distances.append(best_distance)
        
        if route_length(best_neighbor, dist_matrix) < best_distance:
            best_solution = best_neighbor
            best_distance = route_length(best_solution, dist_matrix)
         
            
            if show_plots:
                display_route(points, best_solution, ax)
        
        if tabu_size != -1 and len(tabu_list) >= tabu_size:
            tabu_list.pop(0)
        tabu_list.append(current_solution)
        
        current_solution = best_neighbor

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

    parser = argparse.ArgumentParser(description='Tabu Search for TSP')
    parser.add_argument('--cities', type=int, default=10, help='Number of cities')
    parser.add_argument('--iterations', type=int, default=2000, help='Number of iterations')
    parser.add_argument('--tabu_size', type=int, default=16, help='Tabu list size')
    parser.add_argument('--show_plots', action='store_true', help='Show plots during optimization')

    args = parser.parse_args()

    np.random.seed(42)
    points = np.random.uniform(-1, 1, (args.cities, 2))
    dist_matrix = distance_matrix(points)

    solution, distance, distances = tabu_search(points, dist_matrix, args.tabu_size, args.iterations, args.show_plots)

    if not args.show_plots:
        plot_convergence_curve(distances, title='Tabu Search Convergence Curve')