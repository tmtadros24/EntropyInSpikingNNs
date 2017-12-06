import mnist_loader
import zero_bias_network
import network
import numpy as np
from matplotlib import pyplot as plt
from neuron import *
from ifnetwork import *
#import cPickle as pickle


class RunAutoencoder:
	def __init__(self, NetworkType = 0, layer_sizes=[784,30,5,30,784], batch_size=10, epochs=10, lmbda=1.0):
		training_data, validation_data, test_data,train_ae,val_ae,test_ae = mnist_loader.load_data_wrapper()
		# train autoencoder with no biases
		if NetworkType == 0:
			net_ae = zero_bias_network.Network(layer_sizes, cost=zero_bias_network.QuadraticCost)
			net_ae.large_weight_initializer()
			net_ae.SGD(train_ae, epochs, batch_size, 0.5, lmbda=lmbda)
		else: # train with biases
			net_ae = network.Network(layer_sizes, cost=network.QuadraticCost)
			net_ae.large_weight_initializer()
			net_ae.SGD(train_ae, epochs, batch_size, 0.5, lmbda=lmbda)

		self.autoencoder = net_ae

		self.weights = net_ae.weights
		self.biases  = net_ae.biases

		# Train classifier
		classifier = network.Network([784, 30, 10], cost=network.QuadraticCost)
		classifier.large_weight_initializer()
		classifier.SGD(training_data, epochs*3, batch_size, 3.0, lmbda=lmbda) # learning rate is 3 you can change that if you want

		self.classifier = classifier


	def get_weights(self):
		return self.weights, self.biases

	def get_network(self):
		return self.autoencoder

	def get_classifier(self):
		return self.classifier