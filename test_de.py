import pytest 
import numpy as np
import copy

from differential_evolution import DifferentialEvolution

# CONSTANTS

def rastrigin(array, A=10):
    return A * 2 + (array[0] ** 2 - A * np.cos(2 * np.pi * array[0])) + (array[1] ** 2 - A * np.cos(2 * np.pi * array[1]))

BOUNDS = np.array([[-20, 20], [-20, 20]])
FOBJ = rastrigin


"""
Ваша задача добиться 100% покрытия тестами DifferentialEvolution
Различные этапы тестирования логики разделяйте на различные функции
Запуск команды тестирования:
pytest -s test_de.py --cov-report=json --cov
"""

def test_initialization():
    bounds = np.array([[-20, 20], [-20, 20]])
    population_res = DifferentialEvolution(rastrigin, bounds)
    assert population_res.fobj == rastrigin
    assert np.array_equal(population_res.bounds, bounds)
    assert population_res.mutation_coefficient == 0.8
    assert population_res.crossover_coefficient == 0.7
    assert population_res.population_size == 20
    assert population_res.dimensions == len(bounds)
    assert population_res.a is None

def test_init_population():
    bounds = np.array([[-20, 20], [-20, 20]])
    population_res = DifferentialEvolution(rastrigin, bounds)
    population_res._init_population()
    assert population_res.population.shape == (population_res.population_size, population_res.dimensions)
    assert np.all((population_res.population >= 0) & (population_res.population <= 1))

def test_iterate():
    de_solver = DifferentialEvolution(FOBJ, BOUNDS)
    de_solver._init_population()

    basic_steps_array = [40, 100, 150, 200]
    before = copy.deepcopy(de_solver.fitness)
    for steps in basic_steps_array:
        for _ in range(steps):
            de_solver.iterate()
        assert np.all(before > de_solver.fitness)

if __name__ == "__main__":
    pytest.main(["-s", "test_de.py", "--cov-report=json", "--cov"])