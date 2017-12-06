import numpy as np
from neuron import Neuron
class IFNetwork:
	def __init__(self, T, neurons, weights, biases):
		self.neurons = neurons[1:]
		self.T = 100
		self.dt = 1
		self.time = np.arange(0, self.T, self.dt)
		self.weights = weights
		self.biases = biases

		# create the neurons
		self.network = self.create_neurons()

	def create_neurons(self):
		IF_network = []
		for i in range(len(self.neurons)):
			neuron_list = []
			for j in range(self.neurons[i]):
				neuron_list.append(Neuron())
			IF_network.append(neuron_list)
		return IF_network

	def feedforward(self, spiketrain):
		for i in range(len(self.neurons)):
			current = np.dot(self.weights[i], spiketrain) + self.biases[i]
			spiketrain = np.zeros((len(self.network[i]), 100))
			for h in range(len(self.network[i])):
				spiketrain[h,:] = self.network[i][h].update(current[h])

		return spiketrain