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
       
        test_out = [x[1] for x in test_data]

		# Train classifier
		classifier = network.Network([784, 30, 10], cost=network.QuadraticCost)
		classifier.large_weight_initializer()
		classifier.SGD(training_data, epochs*3, batch_size, 3.0, lmbda=lmbda) # learning rate is 3 you can change that if you want
        #self.weights = classifier.weights
		#self.biases  = classifier.biases
		self.classifier = classifier
        #self.


	def get_weights(self):
		return self.weights, self.biases

	def get_network(self):
		return self.autoencoder

	def get_classifier(self):
		return self.classifier