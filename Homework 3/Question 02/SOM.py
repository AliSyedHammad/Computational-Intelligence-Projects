import numpy as np
from tqdm import tqdm
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from hexalattice import *
import itertools
import random
import json

# generic implementation of Self Organizing Maps
class SOM:
    def __init__(self, map_size: tuple, learning_rate: int, epochs: int)-> None:
        """
        Initialize the SOM with the given parameters and random weights.
        """
        self.map_size = map_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        map_shape = map_size[0], map_size[1], map_size[2]
        self.map = np.random.random(map_shape)

        self.map_radius = max(self.map_size[0], self.map_size[1])/2
        self.time_constant = self.epochs/np.log(self.map_radius)

        self._2d_coordinates_map = self._get_2d_coordinates_grid(self.map_size)

    def train(self, training_data, with_labels=True, with_plot=False):
        """
        Start the training of the SOM with the given input vector and plotiate the map after each epoch.
        """
        if with_labels:
            # Extract labels from each row
            self.labels = [row[0] for row in training_data]
            # removing labels from each row
            training_data = np.array(training_data)[:,1:]
            training_data = np.array(training_data, dtype=float)
        else:
            training_data = np.array(training_data, dtype=float)

        # Normalize the data
        self.training_data = training_data/np.linalg.norm(training_data)
        print(self.training_data.shape)

        print("Started training...")
        for epoch in tqdm(range(self.epochs)):
            # get learning rate
            learning_rate = self._learning_rate_function(time=epoch)
            # get neighborhood radius
            neighborhood_radius = self._neighborhood_radius(time=epoch)
            # iterating over all training data and train the SOM
            for data in self.training_data:
                self._train_on_element(data, learning_rate, neighborhood_radius)
            if with_plot:
                self.plot_map()
            
    
    def _train_on_element(self, element, learning_rate, neighborhood_radius):
        
        distance_map = np.linalg.norm(element - self.map, axis=2)
        # Get index of weight closest to training element
        winner = np.unravel_index(np.argmin(distance_map), distance_map.shape)
        # Get neighborhood function
        neighborhood_function = self.neighbourhood_function(winner, neighborhood_radius)
        # Stack neighborhood function 3 times in z-direction, so weights map can be multiplied by it
        neighborhood_function_replicated = np.dstack(
            tuple([neighborhood_function for _ in range(self.map_size[2])]))

        weight_change = neighborhood_function_replicated * learning_rate * (element - self.map)
        self.map += weight_change
    
    def neighbourhood_function(self, winner, radius):
        """
        Guassian function for the dynamic neighbourhood function. This functions returns a 2D array of the size of the map
        and values in the array are of learning rate multiplied by the guassian function or neighborhood function.
        """
        distance_to_winner = np.linalg.norm(winner - self._2d_coordinates_map, axis=2)
        exponent = -1 * (distance_to_winner / (2 * radius**2))**2
        return np.exp(exponent)

    def _neighborhood_radius(self, time):
        """
        Radius of the neighborhood function for the dynamic neighborhood function. This radius will decay with time.
        """
        return self.map_radius * np.exp(-time/self.time_constant)

    def _learning_rate_function(self, time):
        """
        Learning rate function for the dynamic learning rate. This learning rate will decay with time.
        """
        return self.learning_rate * np.exp(-time/self.epochs)


    def _get_2d_coordinates_grid(self, shape):
        """
        Given a 2 element tuple, return a 2D matrix of grid coordinates.
        Each element of the grid is a point [x, y] where x and y are coordinate values
        """

        y = range(0, shape[0])
        x = range(0, shape[1])

        yx_list = list(itertools.product(y, x))
        yx_vector = np.array(yx_list)
        
        yx_matrix = np.array(yx_vector).reshape((shape[0], shape[1], 2))
        return yx_matrix

    def _add_labels(self):
        """
        Mapping Labels to the best index of SOM grid
        """
        label_indices = []
        for data, label in zip(self.training_data, self.labels):
            distance_map = np.linalg.norm(data - self.map, axis=2)
            # Get index of weight closest to training data
            winner = np.unravel_index(np.argmin(distance_map), distance_map.shape)
            label_indices.append((winner,label))
        return label_indices

    def map_to_colors(self, with_reshape=True):
        """
        Map the weights of the SOM to colors.
        """
        colors = np.zeros((self.map_size[0],self.map_size[1], 3))
        for weight in range(self.map.shape[0]):
            if self.map[weight].shape[1] > 3:
                colors[weight] = self.map[weight][:,1:4]
            elif self.map[weight].shape[1] == 3:
                colors[weight] = self.map[weight]
            else:
                colors[weight] = np.dstack(tuple([self.map[weight] for _ in range(3)]))

        if with_reshape:
            return colors.reshape((self.map_size[0]*self.map_size[1], 3))
        return colors


    def save_map(self, filename):
        """
        Save the map to a file and convert the color format to RGB [0,255]
        """
        colors = self.map_to_colors(with_reshape=False)
        label_indices = self._add_labels()

        info = {}
        for x in label_indices:
            info[x[1]] = [int(i*255) for i in list(colors[x[0][0]][x[0][1]])]

        file = open(filename, 'w')
        json.dump(info, file)
        file.close()
        print("File Saved!")

    def plot_map(self, save=False, filename="map.png"):
        """
        Plots the map of the SOM.
        """
        self.map = np.abs(self.map)
        colors = self.map_to_colors()
        hex_centers, _ = create_hex_grid(nx=self.map_size[0],
                                 ny=self.map_size[1],
                                 do_plot=False)

        x_hex_coords = hex_centers[:, 0]
        y_hex_coords = hex_centers[:, 1]

        plot_single_lattice_custom_colors(x_hex_coords, y_hex_coords,
                                      face_color=colors,
                                      edge_color=colors,
                                      min_diam=1.,
                                    plotting_gap=0,
                                    rotate_deg=0,
                                    line_width=0.5,)

        # Adding labels to SOM Visualization
        if self.labels:
            x_hex_coords = x_hex_coords.reshape((self.map_size[0], self.map_size[1]))
            y_hex_coords = y_hex_coords.reshape((self.map_size[0], self.map_size[1]))
            label_indices = self._add_labels()
            labels_coords = []
            for label in label_indices:
                x =  x_hex_coords[label[0][0]][label[0][1]] - 0.25  # adjusting padding for label text
                y = y_hex_coords[label[0][0]][label[0][1]] - 0.1    # adjusting padding for label text
                l = label[1]
                labels_coords.append((x,y,l))

            taken = []
            for x,y,l in labels_coords:
                if (x,y) not in taken:
                    plt.text(x, y, l, fontsize=7.8, color="white")
                    taken.append((x,y))
        if save:
            plt.savefig('Maps/'+filename)
        plt.show()