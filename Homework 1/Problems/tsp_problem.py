from Problems.problem import Problem
from selection_functions import SelectionFunctions
from EA import *
import random
import numpy as np

class TSPProblem(Problem):
    """
    This class will contain the chromosome representation and  fitness function. This will also handle load data

    Constraints:
     - Not all cities are connected
     - Visit each city atleast once
     - Returns to the starting city
    """
    def __init__(self) -> None:
        self.data = {}
        self.__load_data()
        self.population = self.__generate_initial_population()
        self.fitness_scores = self.fitness_score()

    def __load_data(self):
        """
        This method will load data from the given dataset and convert it into in a suitable
        chromosone representation
        """
        nodes_and_points = {}
        file = open("TSP_dataset/qa194.tsp", "r")
        for line in file.readlines()[7:-1]:
            line = line.split()
            self.data[int(line[0])] = (float(line[1]), float(line[2]))
        file.close()


    def __generate_initial_population(self)-> list:
        """
        Every problem has its own generate_initial_population method 
        so we can have the option of implementing some other logic for it instead of always randomizing
        """
        all_cities = list(self.data.keys())
        random_population= []
        for element in range(POPULATION_SIZE):
            random.shuffle(all_cities)
            # print(all_cities)
            random_population.append(all_cities.copy())
        return random_population

    def crossover(self, parent1: list, parent2: list) -> list:
        """
        This method will perform crossover on the parents and return the child. It selects a random range and selects a sublist from the parents
        and merges them.
        We have eliminated the possibility of wrong chromosomes. 
        """
        crossover_point1 = random.randint(0, len(parent1)-1)
        crossover_point2 = random.randint(crossover_point1, len(parent1)-1)

        child = parent1[crossover_point1:crossover_point2]
        from_parent_2a = parent2[0:crossover_point1]
        from_parent_2b = parent2[crossover_point2:len(parent2)]

        for i in range(len(from_parent_2b)):
            if from_parent_2b[i] in child:
                from_parent_2b[i] = parent1[crossover_point2+i]
        for i in range(len(from_parent_2a)-1,-1,-1):
            if from_parent_2a[i] in child and from_parent_2a[i] not in from_parent_2b:
                from_parent_2a[i] = parent1[i-crossover_point1]
        
        child = from_parent_2a + child + from_parent_2b

        for i in range(len(child)):
            if child.count(child[i]) > 1:
                if parent1[i] not in child:
                    child[i] = parent1[i]
                elif parent2[i] not in child:
                    child[i] = parent2[i]
                else:
                    child[i] = [x for x in parent1 if x not in child][0]
        
        # check if the child is valid
        for each in child:
            assert child.count(each) == 1

        # child = []
        # childP1 = []
        # childP2 = []
        
        # geneA = int(random.random() * len(parent1))
        # geneB = int(random.random() * len(parent1))
        
        # startGene = min(geneA, geneB)
        # endGene = max(geneA, geneB)

        # for i in range(startGene, endGene):
        #     childP1.append(parent1[i])
            
        # childP2 = [item for item in parent2 if item not in childP1]

        # child = childP1 + childP2
        return child





    def mutation(self, child: list, mutation_rate: float()) -> list:
        """
        For the mutation, we will swap two values in the child at mutation_rate probability
        """
        # random_val = random.random()
        # if random_val < mutation_rate:
        #     index1 = random.randint(0, len(child)-1)
        #     index2 = random.randint(0, len(child)-1)
        #     child[index1], child[index2] = child[index2], child[index1]

        point1 = random.randint(0, len(child)-1)
        point2 = random.randint(point1, len(child)-1)

        c_middle = child[point1:point2]
        c_first = child[:point1]
        c_last = child[point2:]

        child = c_first + c_middle[::-1] + c_last
        
        return child

    
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
        total_distance = 0
        for i in range(len(chromosome)-1):
            total_distance += self.__distance(chromosome[i], chromosome[i+1])
        total_distance += self.__distance(chromosome[0], chromosome[-1])
        return total_distance*-1


    def __distance(self, city1: int, city2: int)-> int:
        """
        This method will calculate the distance between two cities
        """
        return ((self.data[city1][0] - self.data[city2][0])**2 + (self.data[city1][1] - self.data[city2][1])**2)**0.5


    