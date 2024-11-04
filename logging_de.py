import logging 


import numpy as np
from datetime import datetime


#База
logger = logging.getLogger(__name__)
# Handler для дебага
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('logging_de.log')
file_handler.setLevel(logging.DEBUG)

# Handler для ошибок
error_handler = logging.FileHandler('errors.log')
error_handler.setLevel(logging.ERROR)

# Формат для данных
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(error_handler)

class DifferentialEvolution:
    # Основные параметры на каждой итерации
    def _log_parameters(self):
        logger.info(f"Population Size: {self.population_size}")
        logger.info(f"Mutation Coefficient: {self.mutation_coefficient}")
        logger.info(f"Crossover Coefficient: {self.crossover_coefficient}")
        logger.info(f"Number of Iterations: {len(self.fitness)}")
        logger.info(f"Best Solution: {self.best}")
        logger.info(f"Best Fitness: {self.fitness[self.best_idx]}")

    # Проверка результатов текущей итерации, не знаю правильно ли вообще это делать внутри класса, но попробуем.
    def _log_results(self):
        if self.fitness[self.best_idx] > 1e-1:
            logger.critical("result above 1e-1 threshold.")
        elif self.fitness[self.best_idx] > 1e-3:
            logger.error("result above 1e-3 threshold.")
        else:
            logger.info("completed successfully.")


    def __init__(self, fobj, bounds, mutation_coefficient=0.8, crossover_coefficient=0.7, population_size=20):


        self.fobj = fobj
        self.bounds = bounds
        self.mutation_coefficient = mutation_coefficient
        self.crossover_coefficient = crossover_coefficient
        self.population_size = population_size
        self.dimensions = len(self.bounds)

        self.a = None
        self.b = None
        self.c = None
        self.mutant = None
        self.population = None
        self.idxs = None
        self.fitness = []
        self.min_bound = None
        self.max_bound = None
        self.diff = None
        self.population_denorm = None
        self.best_idx = None
        self.best = None
        self.cross_points = None

    def _init_population(self):
        self.population = np.random.rand(self.population_size, self.dimensions)
        self.min_bound, self.max_bound = self.bounds.T

        self.diff = np.fabs(self.min_bound - self.max_bound)
        self.population_denorm = self.min_bound + self.population * self.diff
        self.fitness = np.asarray([self.fobj(ind) for ind in self.population_denorm])

        self.best_idx = np.argmin(self.fitness)
        self.best = self.population_denorm[self.best_idx]
    
    def _mutation(self):
        self.a, self.b, self.c = self.population[np.random.choice(self.idxs, 3, replace = False)]
        self.mutant = np.clip(self.a + self.mutation_coefficient * (self.b - self.c), 0, 1)
        return self.mutant
    
    def _crossover(self):
        cross_points = np.random.rand(self.dimensions) < self.crossover_coefficient
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dimensions)] = True
        return cross_points

    def _recombination(self, population_index):

        trial = np.where(self.cross_points, self.mutant, self.population[population_index])
        trial_denorm = self.min_bound + trial * self.diff
        return trial, trial_denorm
    
    def _evaluate(self, result_of_evolution, population_index):
        if result_of_evolution < self.fitness[population_index]:
                self.fitness[population_index] = result_of_evolution
                self.population[population_index] = self.trial
                if result_of_evolution < self.fitness[self.best_idx]:
                    self.best_idx = population_index
                    self.best = self.trial_denorm

    def iterate(self):
    
        for population_index in range(self.population_size):
            self.idxs = [idx for idx in range(self.population_size) if idx != population_index]

            self.mutant = self._mutation()
            self.cross_points = self._crossover()

            self.trial, self.trial_denorm = self._recombination(population_index)
    
            result_of_evolution = self.fobj(self.trial_denorm)

            self._evaluate(result_of_evolution, population_index)

        # Работа logger иначе зачем вообще был весь код в начале класса :)
        self._log_parameters()
        self._log_results()




def rastrigin(array, A=10):
    return A * 2 + (array[0] ** 2 - A * np.cos(2 * np.pi * array[0])) + (array[1] ** 2 - A * np.cos(2 * np.pi * array[1]))


if __name__ == "__main__":



    function_obj = rastrigin
    bounds_array = np.array([[-20, 20], [-20, 20]]), np.array([[-10, 50], [-10, 60]]), np.array([[-0, 110], [-42, 32]])
    steps_array = [40, 100, 200]
    mutation_coefficient_array = [0.5, 0.6, 0.3]
    crossover_coefficient_array = [0.5, 0.6, 0.3]
    population_size_array = [20, 30, 40, 50, 60]

    for bounds in bounds_array:
        for steps in steps_array:
            for mutation_coefficient in mutation_coefficient_array:
                for crossover_coefficient in crossover_coefficient_array:
                    for population_size in population_size_array:

                        de_solver = DifferentialEvolution(function_obj, bounds, mutation_coefficient=mutation_coefficient, crossover_coefficient=crossover_coefficient, population_size=population_size)

                        de_solver._init_population()

                        for _ in range(steps):
                            de_solver.iterate()