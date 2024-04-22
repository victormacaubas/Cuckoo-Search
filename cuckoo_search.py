import numpy as np
import math


class CuckooSearch:
    def __init__(self, problem, population_size, max_iterations):
        self.problem = problem
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.alpha = 1.0  # Scaling factor for Lévy flights
        self.pa = 0.25  # Probability of discovering an alien egg
        self.nests = []
        self.fitness = []

    def levy_flight(self, beta=1.5):
        """ Generate step sizes via Lévy flight. """
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
                 (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, 1)
        v = np.random.normal(0, 1, 1)
        step = u / abs(v) ** (1 / beta)
        return step

    def initialize_population(self):
        self.nests = np.random.uniform(self.problem['bounds'][0], self.problem['bounds'][1],
                                       (self.population_size, len(self.problem['variables'])))
        self.fitness = np.array([self.problem['function'](nest)
                                for nest in self.nests])

    def get_best_nest(self):
        best_idx = np.argmin(self.fitness)
        return self.nests[best_idx], self.fitness[best_idx]

    def abandon_worse_nests(self):
        n_abandon = int(self.pa * self.population_size)
        worse_indices = np.argsort(self.fitness)[-n_abandon:]
        for idx in worse_indices:
            self.nests[idx] = np.random.uniform(self.problem['bounds'][0], self.problem['bounds'][1],
                                                (len(self.problem['variables'])))
            self.fitness[idx] = self.problem['function'](self.nests[idx])

    def run(self):
        self.initialize_population()
        best_nest, best_fitness = self.get_best_nest()

        for iteration in range(self.max_iterations):
            for i in range(self.population_size):
                new_nest = self.nests[i] + self.alpha * self.levy_flight()
                new_fitness = self.problem['function'](new_nest)

                # Randomly choose a nest to replace (but not the best one)
                j = np.random.randint(0, self.population_size)
                if new_fitness < self.fitness[j]:
                    self.nests[j] = new_nest
                    self.fitness[j] = new_fitness
                    if new_fitness < best_fitness:
                        best_nest = new_nest
                        best_fitness = new_fitness

            # Abandon a fraction of the worst nests
            self.abandon_worse_nests()

            # Find the best nest after potential changes
            current_best, current_fitness = self.get_best_nest()
            if current_fitness < best_fitness:
                best_nest = current_best
                best_fitness = current_fitness

        return best_nest, best_fitness
