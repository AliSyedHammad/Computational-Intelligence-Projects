import numpy as np

class Ant:

    ALPHA = 3
    BETA = 5
    GAMMA = 0.25

    def __init__(self, nodes: dict, warehouse, capacity, pheromone_map, adjacency_map):
        self.capacity = capacity
        self.warehouse = warehouse
        self.nodes : list = nodes
        self._no_of_nodes = len(nodes)
        self.visited = [warehouse]
        self.total_path_cost = 0
        self.current_city = warehouse
        self.pheromone_map = pheromone_map
        self.adjacency_map = adjacency_map
        
    
    def add_to_visited(self, node_i):
        self.visited.append(node_i)
    
    def has_visited(self, node_i) -> bool:
        return node_i in self.visited
    
    def has_compl_tour(self) -> bool:
        return len(set(self.visited)) == self._no_of_nodes

    def select_next_city(self):
        available_cities = list(set(self.nodes).difference(set(self.visited)))

        if not available_cities:
            return None
        
        scores = [pow(self.pheromone_map[[*self.nodes].index(self.current_city)][[*self.nodes].index(city)], Ant.ALPHA) *
                    pow(1 / (self.adjacency_map[[*self.nodes].index(self.current_city)][[*self.nodes].index(city)] + 1e-10), Ant.BETA)
                    for city in available_cities]
        denominator = sum(scores)
        probabilities = [score / denominator for score in scores]
        
        next_city = np.random.choice(available_cities, p=probabilities)

        return next_city

    def get_routes(self):
        lst = self.visited[1:]
        routes = []
        
        try:
            while i := lst.index(self.warehouse):
                routes.append([self.warehouse]+lst[:i+1])
                lst=lst[i+1:]
        except ValueError:
            routes.append([self.warehouse]+lst+[self.warehouse])
            return routes


    def move_to_city(self, city, capacity):
        self.total_path_cost += self.adjacency_map[[*self.nodes].index(self.current_city)][[*self.nodes].index(city)]
        self.visited.append(city)
        self.capacity -= capacity
        self.current_city = city



def calculate_distance(node1, node2):
    x1 = node1[0]
    x2 = node2[0]
    y1 = node1[1]
    y2 = node2[1]

    return np.sqrt((x2-x1)**2 + (y2-y1)**2)
