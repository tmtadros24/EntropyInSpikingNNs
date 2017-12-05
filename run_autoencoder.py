
import mnist_loader
import network
import numpy as np
from matplotlib import pyplot as plt
from neuron import *
from ifnetwork import *
import cPickle as pickle

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


training_data, validation_data, test_data,train_ae,test_ae = mnist_loader.load_data_wrapper()
spike_trains = generate_spike_trains(test_ae[0:100][:], 100, 1)

''' plot spike frequency over time
m = np.zeros(784)
# Number of Spikes
n=10
plt.figure(figsize=(10, 4))
plt.autoscale(False)
for i in range(10):
	m = np.zeros(784)
	# display original
	ax = plt.subplot(2, n, i + 1)
	j = 0
	while j < i*10 + 1:
		m = np.add(m, spike_trains[1,:,j])
		j += 1
	print np.max(m)
	plt.imshow(m.reshape(28,28) / 100.0, vmin=0, vmax=1)
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
plt.show()
'''

# Train autoencoder
net_ae = network.Network([784, 30, 5, 30, 784], cost=network.QuadraticCost)
net_ae.large_weight_initializer()
net_ae.SGD(train_ae, 10, 10, 0.5)

# Save weights and biases

weights = net_ae.weights
biases = net_ae.biases

pickle.dump(weights, open( "weights_normal.p", "wb" ) )
pickle.dump(biases, open( "biases_normal.p", "wb" ) )

ifnetwork = IFNetwork(100, [784, 30, 5, 30, 784], weights, biases)
output_spikes = ifnetwork.feedforward(spike_trains[1])

spike = spike_trains[1]
n=10

plt.figure(figsize=(20, 4))
plt.autoscale(False)
for i in range(10):

	m = np.zeros(784)
	# display original
	ax = plt.subplot(2, n, i + 1)
	j = 0
	while j < i*10 + 1:
		m = np.add(m, spike_trains[1,:,j])
		j += 1
	plt.imshow(m.reshape(28,28) / 100.0, vmin=0, vmax=1)
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)


	m = np.zeros(784)
	# display original
	ax = plt.subplot(2, n, i + 1+n)
	j = 0
	while j < i*10 + 1:
		m = np.add(m, output_spikes[:,j])
		j += 1
	print np.max(m)
	plt.imshow(m.reshape(28,28) / 10.0, vmin=0, vmax=1)
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

plt.show()
