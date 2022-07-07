from Problems.problem import Problem
from selection_functions import SelectionFunctions
from EA import *
import random


class KnapsackProblem(Problem):
    """
    This class will contain the chromosome and the fitness function
    """
    def __init__(self) -> None:
        self.data = {}
        self.weight_limit=-1
        self.__load_data()
        self.population = self.__generate_initial_population()
        self.fitness_scores = self.fitness_score()

    def __load_data(self):
        """
        This method will load data from the given dataset and convert it into in a suitable
        chromosone representation
        """
        nodes_and_points = {}
        file = open("KP_Dataset/low-dimensional/f2_l-d_kp_20_878", "r")
        for i, line in enumerate(file.readlines()):
            if i == 0:
                line = line.split()
                self.weight_limit = int(line[1])
            else:
                line = line.split()
                self.data[i-1] = (int(line[0]), int(line[1]))
        file.close()

    def __generate_initial_population(self)-> list:
        """
        Every problem has its own generate_initial_population method 
        so we can have the option of implementing some other logic for it instead of always randomizing
        """
        random_population = []
        for element in range(POPULATION_SIZE):
            all_items = list(self.data.keys())
            random.shuffle(all_items)
            # print(all_cities)
            knapsack = []
            current_weight = 0
            while(current_weight < self.weight_limit):
                item = all_items.pop(0)
                knapsack.append(item)
                current_weight += self.data[item][1]
            knapsack.pop()
            random_population.append(knapsack)
        return random_population
    
    
    def fitness_score(self)-> list:
        """
        This method will calculate the fitness score of the population
        """
        fitness_score = []
        for chromosome in self.population:
            fitness_score.append(self.__fitness_function(chromosome))
        return fitness_score

    def __fitness_function(self, chromosome: list)-> int:
        """
        This method will calculate the fitness of the chromosome
        """
        total_profit = 0
        for item in chromosome:
            total_profit += self.data[item][0]
        return total_profit

    
    def crossover(self, parent1: list, parent2: list) -> list:
        """
        This method will perform crossover on the parents and return the child. It selects a random range and selects a sublist from the parents
        and merges them.
        We have eliminated the possibility of wrong chromosomes. 
        """
        crossover_point1 = random.randint(0, len(parent1)-1)
        crossover_point2 = random.randint(0, len(parent2)-1)
        child = []
        parent_flag = True
        count = 0
        while self.calculate_weight(child) < self.weight_limit and count < 100:
            if parent_flag:
                if not parent1[crossover_point1%len(parent1)] in child:
                    child.append(parent1[crossover_point1%len(parent1)])
                parent_flag = not parent_flag
                crossover_point1+=1
            else:
                if not parent2[crossover_point2%len(parent2)] in child:
                    child.append(parent2[crossover_point2%len(parent2)])
                parent_flag = not parent_flag
                crossover_point2+=1
            count+=1
        child.pop()
        return child

    def calculate_weight(self, chromosome: list) -> list :
        """
        This method will calculate the weight of the knapsack
        """
        total_weight = 0
        for item in chromosome:
            total_weight += self.data[item][1]
        return total_weight


    def mutation(self, child: list, mutation_rate: float()) -> list:
        """
        For the mutation, we will swap two values in the child at mutation_rate probability
        """
        random_val = random.random()
        if random_val < mutation_rate:
            index = random.randint(0, len(child)-1)
            child.pop(index)
            current_weight = self.calculate_weight(child)
            new_index = random.randint(0, len(self.data)-1) 
            # TODO: need to reduce while
            count = 0
            while (current_weight+self.data[new_index][1] > self.weight_limit or new_index in child) and count < 100: 
                new_index = random.randint(0, len(self.data)-1)
                count += 1
            child.append(new_index)

        return child
        
