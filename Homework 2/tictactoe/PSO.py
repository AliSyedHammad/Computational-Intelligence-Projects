import random
import numpy as np
import board
import tictactoe
from tqdm import tqdm

class Particle:
    def __init__(self, position, velocity, fitness=None):
        self.position = position
        self.velocity = velocity
        self.fitness = fitness
        self.pbest_position = self.position
        self.pbest_fitness = fitness

    def copy(self):
        return Particle(self.position, self.velocity, self.fitness)

class ParticleSwarmOptimizer:
    """
    Implementing PCO algorithm using Neural Network. We are considering every particle to be 
    a set of weights for the neural network.
    """
    def __init__(self, num_particles, num_dimensions, fitness_function,max_velocity=10, max_position=10):
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.max_velocity = max_velocity
        self.max_position = max_position
        self.c1 = 0.6
        self.c2 = 0.6
        self.w = 1.3
        self.particles = []
        self.global_best = None
        self.fitness_function = fitness_function
        self.best_history = []
        self._init_swarm()

    def _init_swarm(self):
        print("Initializing swarm...")
        for i in range(self.num_particles):
            position = np.random.uniform(0, self.max_position, self.num_dimensions)
            velocity = np.random.uniform(0, self.max_velocity, self.num_dimensions)
            fitness = self.fitness_function(position)
            particle = Particle(position, velocity, fitness)
            self.particles.append(particle)
            self.global_best = particle.copy()
        self.best_history.append(self.global_best.fitness)

    def optimize(self, iterations):
        print("Optimizing swarm...")
        for _ in range(iterations):
            self.best_history.append(self.global_best.fitness)
            for particle in self.particles:
                print("Global best: ", self.global_best.fitness)
                particle.velocity = (self.w * particle.velocity) + (self.c1 * random.random() * (particle.pbest_position - particle.position)) + (self.c2 * random.random() * (self.global_best.position - particle.position))
                particle.position = particle.position + particle.velocity
                particle.fitness = self.fitness_function(particle.position)
                if particle.fitness > self.global_best.fitness:
                    self.global_best = particle.copy()
                if particle.fitness > particle.pbest_fitness:
                    particle.pbest_fitness = particle.fitness
                    particle.pbest_position = particle.position

    def get_global_best(self):
        return self.global_best

    def get_best_history(self):
        return self.best_history