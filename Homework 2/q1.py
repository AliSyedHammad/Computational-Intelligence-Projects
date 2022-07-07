import xmltodict
from pprint import pprint
from ant import Ant, calculate_distance
import numpy as np
import matplotlib.pyplot as plt

ALPHA = 2
BETA = 2
GAMMA = 0.5

def add_pheromone(pheromone_map, node0, node1, fitness):
    new_value = max(round((1 - GAMMA) * pheromone_map[node0][node1], 2) + 1/fitness, 1e-6)
    pheromone_map[node0][node1] = new_value
    pheromone_map[node1][node0] = new_value

def update_pheromone_map(pheromone_map, routes, fitness, nodes):
    ordered = [item for sublist in routes for item in sublist]
    
    for i in range(len(ordered)-1):
        add_pheromone(pheromone_map, [*nodes].index(ordered[i]), [*nodes].index(ordered[i+1]), fitness)


def main():
    
    population = 10
    iterations = 100

    with open('A-n80-k10.xml', 'r', encoding='utf-8') as file:
        my_xml = file.read()
    
    my_dict = xmltodict.parse(my_xml, dict_constructor=dict)

    nodes = my_dict["instance"]["network"]["nodes"]["node"]
    nods = dict()

    for i in range(len(nodes)):
        if nodes[i]["@type"] == "0":
            warehouse = eval(nodes[i]["@id"])
        nods[eval(nodes[i]["@id"])] = eval(nodes[i]["cx"]), eval(nodes[i]["cy"])

    capacity = eval(my_dict["instance"]["fleet"]["vehicle_profile"]["capacity"])
    requests = my_dict["instance"]["requests"]["request"]
    reqs = dict()

    for request in requests:
        reqs[eval(request["@node"])] = eval(request["quantity"])
    
    pheromone_map = [[1e-6 for _ in nods] for _ in nods]
    adjacency_map = [[calculate_distance(node0, node) for node0 in nods.values()] for node in nods.values()]


    best_fitness = 1e-10
    best_routes = []
    best_history = []

    for i in range(iterations):
        ants = [Ant(nods, warehouse, capacity, pheromone_map, adjacency_map) for _ in range(population)]

        while np.sum([ants[i].has_compl_tour() for i in range(population)]) < population:
            for ant in ants:
                new_city = ant.select_next_city()
                if new_city:    
                    if ant.capacity < reqs[new_city]:
                        ant.move_to_city(warehouse, ant.capacity - capacity)

                    else:
                        ant.move_to_city(new_city, reqs[new_city])
                else:
                    pass
        
        fitnesses = []


        for ant in ants:
            update_pheromone_map(pheromone_map, ant.get_routes(), ant.total_path_cost, nods)
            fitnesses.append(ant.total_path_cost)

        this_best = 1 / ants[np.argmin(fitnesses)].total_path_cost
        best_fitness = this_best if this_best > best_fitness else best_fitness 
        best_history.append(best_fitness)
        best_routes = ants[np.argmin(fitnesses)].get_routes() if this_best > best_fitness else best_routes
        update_pheromone_map(pheromone_map, best_routes, capacity/best_fitness, nods)

        # print(pheromone_map)
        print("Best Cost this generation:", 1/this_best)
        print("Best Fitness this generation:", this_best)
        print("Best Cost Overall:", 1/best_fitness)
        print()
    
    plt.plot(best_history)
    plt.ylabel("Best Fitness")
    plt.xlabel("Iteration")
    plt.show()


if __name__ == "__main__":
    main()