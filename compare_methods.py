import subprocess
import argparse

def run_algorithm(script, args):
    subprocess.run(['python', script] + args, capture_output=False, text=True)
    #return result.stdout

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare optimization methods for TSP')
    parser.add_argument('--cities', type=int, default=10, help='Number of cities')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations')
    parser.add_argument('--show_plots', action='store_true', help='Show plots during optimization')

    args = parser.parse_args()
    
    algorithms = [
        'hill_climbing.py',
        'simulated_annealing.py',
        'tabu_search.py',
        'genetic_algorithm.py',
        'brute_force.py'
    ]

    algorithms_args = [
        [f'--cities', str(args.cities), f'--iterations', str(args.iterations)],
        [f'--cities', str(args.cities), f'--iterations', str(args.iterations)],
        [f'--cities', str(args.cities), f'--iterations', str(args.iterations)],
        [f'--cities', str(args.cities), f'--max_generations', str(args.iterations)],
        [f'--cities', str(args.cities)]
    ]
    
    for i, algorithm in enumerate(algorithms):
        print(f'Running {algorithm}...')
        run_algorithm(algorithm, algorithms_args[i])