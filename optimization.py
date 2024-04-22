import numpy as np

class Optimization:
    def __init__(self, dimension, lower_bound, upper_bound):
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
    def random_solution(self):
        """Generate a random solution within the specified bounds."""
        return self.lower_bound + np.random.rand(self.dimension) * (self.upper_bound - self.lower_bound)

class Rastrigin(Optimization):
    def test_function(self, x):
        A = 10
        return A * self.dimension + np.sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

class Rosenbrock(Optimization):
    def test_function(self, x):
        return np.sum([100 * (x[i + 1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(self.dimension - 1)])

class Ackley(Optimization):
    def test_function(self, x):
        a = 20
        b = 0.2
        c = 2 * np.pi
        sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(x**2) / self.dimension))
        cos_term = -np.exp(np.sum(np.cos(c * x) / self.dimension))
        return a + np.exp(1) + sum_sq_term + cos_term

class Sphere(Optimization):
    def test_function(self, x):
        return np.sum(x**2)