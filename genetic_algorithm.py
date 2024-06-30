import random
import time
import numpy as np
import matplotlib.pyplot as plt
from utils import distance_matrix, route_length, display_route, plot_convergence_curve

def initial_population(n_cities, population_size):
    return [random_solution(n_cities) for _ in range(population_size)]

def random_solution(n_cities):
    permutation = list(range(n_cities))
    random.shuffle(permutation)
    return permutation

def select_parents_top_half(population):
    half_population_size = len(population) // 2
    top_half = population[:half_population_size]
    parent1 = random.choice(top_half)
    parent2 = random.choice(top_half)
    return parent1, parent2

def tournament_selection(population, dist_matrix, tournament_size=5):
    tournament = random.sample(population, tournament_size)
    tournament.sort(key=lambda perm: route_length(perm, dist_matrix))
    return tournament[0]

def select_parents(population, dist_matrix, selection_method):
    if selection_method == 'tournament':
        parent1 = tournament_selection(population, dist_matrix)
        parent2 = tournament_selection(population, dist_matrix)
    else:
        parent1, parent2 = select_parents_top_half(population)
    return parent1, parent2

def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[start:end+1] = parent1[start:end+1]
    
    fill_index = 0
    for i in range(size):
        if child[i] == -1:
            while parent2[fill_index] in child:
                fill_index += 1
            child[i] = parent2[fill_index]
    
    return child

def crossover_alternate(parent1, parent2):
    size = len(parent1)
    child = [-1] * size
    
    for i in range(size):
        if i % 2 == 0:
            child[i] = parent1[i]
        else:
            child[i] = parent2[i]
    
    return fix_child(child)

def fix_child(child):
    size = len(child)
    seen = set()
    missing = iter(set(range(size)) - set(child))
    
    for i in range(size):
        if child[i] in seen or child[i] == -1:
            child[i] = next(missing)
        seen.add(child[i])
    
    return child

def mutate(permutation, mutation_rate):
    if random.random() < mutation_rate:
        a, b = random.sample(range(len(permutation)), 2)
        permutation[a], permutation[b] = permutation[b], permutation[a]
    return permutation

def mutate_reverse(permutation, mutation_rate):
    if random.random() < mutation_rate:
        start, end = sorted(random.sample(range(len(permutation)), 2))
        permutation[start:end+1] = reversed(permutation[start:end+1])
    return permutation

def genetic_algorithm(points, dist_matrix, population_size, elite_size, mutation_rate, max_generations, crossover_method, mutation_method, selection_method, termination_condition, max_stagnant_generations, show_plots=True):
    population = initial_population(len(points), population_size)
    generation = 0
    best_distance = float('inf')
    best_solution = None
    distances = []
    stagnant_generations = 0

    if show_plots:
        fig, ax = plt.subplots()
        plt.ion()

    start_time = time.time()

    while generation < max_generations:
        population.sort(key=lambda perm: route_length(perm, dist_matrix))
        current_best_distance = route_length(population[0], dist_matrix)
        if current_best_distance < best_distance:
            best_solution = population[0]
            best_distance = current_best_distance
            stagnant_generations = 0
        else:
            stagnant_generations += 1
        
        distances.append(best_distance)

        if termination_condition == 'stagnation' and stagnant_generations >= max_stagnant_generations:
            break

        next_population = population[:elite_size]

        while len(next_population) < population_size:
            parent1, parent2 = select_parents(population, dist_matrix, selection_method)
            if crossover_method == 'alternate':
                child = crossover_alternate(parent1, parent2)
            else:
                child = crossover(parent1, parent2)
            if mutation_method == 'reverse':
                child = mutate_reverse(child, mutation_rate)
            else:
                child = mutate(child, mutation_rate)
            next_population.append(child)

        population = next_population
        generation += 1

        if show_plots:
            display_route(points, best_solution, ax)

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

    parser = argparse.ArgumentParser(description='Genetic Algorithm for TSP')
    parser.add_argument('--cities', type=int, default=10, help='Number of cities')
    parser.add_argument('--population_size', type=int, default=100, help='Population size')
    parser.add_argument('--elite_size', type=int, default=20, help='Elite size')
    parser.add_argument('--mutation_rate', type=float, default=0.01, help='Mutation rate')
    parser.add_argument('--max_generations', type=int, default=500, help='Maximum number of generations')
    parser.add_argument('--crossover_method', type=str, default='default', choices=['default', 'alternate'], help='Crossover method')
    parser.add_argument('--mutation_method', type=str, default='swap', choices=['swap', 'reverse'], help='Mutation method')
    parser.add_argument('--selection_method', type=str, default='top_half', choices=['top_half', 'tournament'], help='Selection method')
    parser.add_argument('--termination_condition', type=str, default='generations', choices=['generations', 'stagnation'], help='Termination condition')
    parser.add_argument('--max_stagnant_generations', type=int, default=100, help='Maximum number of stagnant generations for termination')
    parser.add_argument('--show_plots', action='store_true', help='Show plots during optimization')

    args = parser.parse_args()

    np.random.seed(42)
    points = np.random.uniform(-1, 1, (args.cities, 2))
    dist_matrix = distance_matrix(points)

    solution, distance, distances = genetic_algorithm(
        points, dist_matrix, args.population_size, args.elite_size,
        args.mutation_rate, args.max_generations, args.crossover_method,
        args.mutation_method, args.selection_method, args.termination_condition,
        args.max_stagnant_generations, show_plots=args.show_plots
    )

    if not args.show_plots:
        plot_convergence_curve(distances, title='Genetic Algorithm Convergence Curve')
    print(f"Execution time: {execution_time:.4f} seconds")