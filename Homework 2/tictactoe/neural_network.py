import numpy as np
from PSO import ParticleSwarmOptimizer
from board import *
from tictactoe import *

# Neural Network with Partcle Swarm Optimization Training
class NeuralNetwork:

    def __init__(self, num_inputs, num_hidden, num_outputs):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.init_weights()

    def init_weights(self):
        self.weights_ih = np.random.uniform(-1, 1, (self.num_hidden, self.num_inputs))
        self.weights_ho = np.random.uniform(-1, 1, (self.num_outputs, self.num_hidden))

    def update_weights(self, ih_weights, ho_weights):
        self.weights_ih = ih_weights
        self.weights_ho = ho_weights

    def feedforward(self, inputs):
        """
        Feedforward the neural network
        """
        hidden_inputs = np.dot(self.weights_ih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.weights_ho, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

    def _encodeBoard(self, board):
        """
        This method is used for encoding the board into a vector for the neural network
        """
        encodedBoard = []
        for i in range(9):
            if board[i] == 'X':
                encodedBoard.append(1)
            elif board[i] == 'O':
                encodedBoard.append(-1)
            else:
                encodedBoard.append(0)
        return encodedBoard

    def predict(self, board, playerLetter):
        """
        Predict the output of the neural network
        """
        encodeBoard = self._encodeBoard(board)
        output = list(self.feedforward(encodeBoard))
        return output.index(max(output)) + 1

    def activation_function(self, x):
        """
        Using ReLU function as activation function
        """
        return np.maximum(0, x)
        

    def calculate_loss(self, weights):
        """
        This method is used for calculating the loss of the neural network and also for the fitness of the particle.
        I am using multiple agents provided in the tictactoe.py file to train the neural network.
        """
        ih_weights, ho_weights = self._reshapeWeights(weights)
        self.update_weights(ih_weights, ho_weights)
        (W1,L1,T1) = PlayMultiple(500,self.predict,getIntelligentMove, visibleMode=False)  #Simulate 500 matches with AI agent
        (W2,L2,T2) = PlayMultiple(500,self.predict,getNaiveMove, visibleMode=False)  #Simulate 500 matches with Naive agent
        (W3,L3,T3) = PlayMultiple(500,self.predict,getUnintelligentMove, visibleMode=False)  #Simulate 500 matches with unIntelligent agent

        return (W1+W2+W3+T1+T2+T3)

    def optimizeUsingPSO(self, num_particles=500, iterations=100, max_position=3, max_velocity=1):
        """
        This method is used for optimizing the weights of the neural network using Particle Swarm Optimization
        """
        dimension = len(self.weights_ho.flatten())+len(self.weights_ih.flatten())
        pso = ParticleSwarmOptimizer(
            num_particles=num_particles, 
            num_dimensions=dimension, 
            fitness_function=self.calculate_loss,
            max_velocity=max_velocity, 
            max_position=max_position)
        
        pso.optimize(iterations=iterations)
        weights = pso.global_best.position

        self.best_history = pso.get_best_history()
        ih_weights, ho_weights = self._reshapeWeights(weights)
        self.update_weights(ih_weights, ho_weights)

    def _reshapeWeights(self, weights):
        """
        This method is used for reshaping the weights of the neural network
        """
        ih_weights = weights[:len(self.weights_ih.flatten())]
        ho_weights = weights[len(self.weights_ih.flatten()):]
        ih_weights = np.array(ih_weights).reshape(self.weights_ih.shape)
        ho_weights = np.array(ho_weights).reshape(self.weights_ho.shape)
        return ih_weights, ho_weights

