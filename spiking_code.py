import cPickle as pickle
import mnist_loader
import network
import numpy as np
from matplotlib import pyplot as plt
from neuron import *
from ifnetwork import *

def generate_spike_trains(D, T, dt):
	M = T/dt
	spike_trains = np.zeros((50000, 784, M))
	for i in range(M):
		X = np.random.uniform(size=M)
		for j in range(len(D)):
			for k in range(784):
				if D[j][0][k] * dt > X[i]:
					if i > 0 and spike_trains[j][k][i-1] == 0:
						spike_trains[j,k,i] = 1

	return spike_trains


'''
Takes in weights and biases and returns reconstructed images from the spiking neural network.
'''
def feedforward_spiking_network(weights, biases):
	assert len(weights) == len(biases) == 4

	training_data, validation_data, test_data, train_ae,_, test_ae = mnist_loader.load_data_wrapper()
	spike_trains = generate_spike_trains(test_ae[:][:], 100, 1)
	output_spike_train = np.zeros((len(test_ae), 784, 100)) # 10000 x 784 x 100 - 100 is number of time steps
	for i in range(len(test_ae)):
		ifnetwork = IFNetwork(100, [784, 30, 5, 30, 784], weights, biases) # biases should be all 0's ideally
		spike_train = spike_trains[i]
		output_spike_train[i] = ifnetwork.feedforward(spike_train)

	return spike_train, output_spike_train # input to network, output from network

'''
Takes in a classifier and list of imagesand computes the accuracy on the test set
'''
def classication(classification_network, data, labels):
	assert len(data) == len(labels)
	accuracy = 0.0
	for i in range(len(data)):
		image = data[i]
		output = classification_network.feedforward(image)

		ground_truth = labels[i]
		if output == ground_truth:
			accuracy += 1.0

	return accuracy / len(test_ae)


'''
Trains a classifier based on training data - we may want training data to be reconstructed images and not just the original training set
'''
def train_classifier(training_data):
	# Train autoencoder
	net_ae = network.Network([784, 30, 10], cost=network.QuadraticCost)
	net_ae.large_weight_initializer()
	net_ae.SGD(training_data, 10, 10, 0.5)

	return net_ae

# plots images and reconstructions if applicable - y x 784 vectors where y is number of images to plot
def plot_images(images, reconstructions=None):
	if reconstructions is not None:
		assert len(images) == len(reconstructions)
		n = len(images)
		plt.figure(figsize=(20, 4))
		plt.autoscale(False)
		for j in range(len(images)):
			ax = plt.subplot(2, n, j+1)
			plt.imshow(images[j].reshape(28,28), vmin=0, vmax=1)
			plt.gray()

			ax = plt.subplot(2,n,j+1+n)
			plt.imshow(reconstructions[j].reshape(28,28), vmin=0, vmax=1)
			plt.gray()

	else:
		n = len(images)
		plt.figure(figsize=(20, 4))
		plt.autoscale(False)
		for j in range(len(images)):
			ax = plt.subplot(2, n, j+1)
			plt.imshow(images[j].reshape(28,28), vmin=0, vmax=1)
			plt.gray()

	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	plt.show()







