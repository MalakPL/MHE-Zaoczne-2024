# MHE-Zaoczne-2024

Komiwojażer (z ang. Traveling Salesman Problem, TSP) to klasyczny problem optymalizacyjny z dziedziny informatyki i badań operacyjnych. Polega on na znalezieniu najkrótszej możliwej trasy, która odwiedza każde miasto dokładnie raz i wraca do punktu wyjścia.

# Metody Optymalizacji Problemów Komiwojażera

To repozytorium zawiera implementacje różnych algorytmów optymalizacyjnych do rozwiązania Problemów Komiwojażera (TSP). Dołączone algorytmy to:

- Przegląd zupełny
- Wspinaczka
- Wyżarzanie
- Przeszukiwanie tabu
- Algorytm genetyczny

Każdy algorytm jest zaimplementowany w osobnym pliku Pythona. Repozytorium zawiera także skrypt porównujący efektywność wszystkich metod.

## Struktura plików

- `brute_force.py`: Implementacja algorytmu przeglądu zupełnego.
- `hill_climbing.py`: Implementacja algorytmu wspinaczkowego.
- `simulated_annealing.py`: Implementacja algorytmu wyżarzania.
- `tabu_search.py`: Implementacja algorytmu przeszukiwania tabu.
- `genetic_algorithm.py`: Implementacja algorytmu genetycznego.
- `utils.py`: Zestaw narzędzi wspomagających (np. obliczanie długości trasy, wyświetlanie trasy, itp.).
- `compare_methods.py`: Skrypt do porównywania efektywności różnych metod optymalizacji.

## Wymagania

- Python 3.6 lub nowszy
- Biblioteki: numpy, matplotlib, scipy

Można je zainstalować za pomocą polecenia:
```bash
pip install numpy matplotlib scipy
```

## Jak używać

### Uruchamianie poszczególnych algorytmów

Każdy z algorytmów można uruchomić niezależnie, korzystając z poniższych poleceń:

#### Przegląd zupełny
```bash
python brute_force.py --cities 10 --show_plots
```

#### Wspinaczka
```bash
python hill_climbing.py --cities 10 --iterations 100 --show_plots
```

#### Wyżarzanie
```bash
python simulated_annealing.py --cities 10 --iterations 10000 --show_plots
```

#### Przeszukiwanie tabu
```bash
python tabu_search.py --cities 10 --iterations 2000 --tabu_size 16 --show_plots
```

#### Algorytm genetyczny
```bash
python genetic_algorithm.py--cities 20 --population_size 100 --elite_size 20 --mutation_rate 0.01 --max_generations 500 --crossover_method default --mutation_method swap --termination_condition stagnation --max_stagnant_generations 50 --show_plots
```

### Porównywanie metod

Aby porównać efektywność wszystkich metod, można użyć skryptu `compare_methods.py`:
```bash
python compare_methods.py --cities 10 --iterations 100 --show_plots
```

Skrypt ten uruchomi każdą z metod optymalizacji z zadanymi parametrami i wyświetli porównanie wyników.
