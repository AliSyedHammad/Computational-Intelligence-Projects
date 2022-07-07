from EA import *
from Problems.tsp_problem import TSPProblem
from Problems.knapsack_problem import KnapsackProblem
from Problems.problem import Problem
from Problems.graph_coloring_problem import GC
import numpy as np
from matplotlib import pyplot as plt    

plt.rcParams['text.usetex'] = True
plt.rcParams["figure.figsize"] = (10,8)



def main():
    # instantiate a problem
    # TSP = TSPProblem()
    gc = GC()
    # knapsack = KnapsackProblem()
    # # applying the EA to the problem
    ea = EA(gc)
    best_fitness_scores = []
    averaga_fitness_scores = []

    # file = open("Results/TSP/results_BT_and_Truncation.txt", "w")

    for i in range(GENERATIONS):
        # file.write("{},{},{}\n".format(ea.generation, abs(ea.best_fitness_score()), abs(ea.averaga_fitness_score())))
        print(ea.generation, abs(ea.best_fitness_score()), len(ea.population))

        # evolutionary process
        ea.generate_offspring(Selection.RankBasedSelection)
        ea.evaluate_population(Selection.BinaryTournament)

        # append the best fitness score to the list for plotting
        best_fitness_scores.append(abs(ea.best_fitness_score()))
        averaga_fitness_scores.append(abs(ea.averaga_fitness_score()))

    print("Fittest canditate: ", ea.population[ea.fitness_scores.index(ea.best_fitness_score())])

    plt.plot(best_fitness_scores, label='Best Fitness Score',)
    plt.plot(averaga_fitness_scores, label='Average Fitness Score',)
    plt.annotate("{}".format(round(best_fitness_scores[-1],3)),
                    (len(best_fitness_scores)-1,best_fitness_scores[-1]), 
                    textcoords="offset points",
                    xytext=(0,10), 
                    ha='center',
                    arrowprops=dict(arrowstyle="->", color='green')
                    )
    plt.annotate("{}".format(round(averaga_fitness_scores[-1],3)),
                    (len(averaga_fitness_scores)-1,averaga_fitness_scores[-1]), 
                    textcoords="offset points",
                    xytext=(0,-20), 
                    ha='center',
                    arrowprops=dict(arrowstyle="->", color='green')
                    )
    plt.title(r'RBS and BT')
    plt.xlabel(r'Generations')
    plt.ylabel(r'Fitness Score')
    plt.legend()
    plt.show()


main()
