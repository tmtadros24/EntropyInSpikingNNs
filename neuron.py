import numpy as np
#Gerstner electrical equivalent of the integrate and fire model , please refer to his literature for the electrical models 

from numpy import *
from pylab import *

class Neuron:
	def __init__(self):
		self.T = 100
		self.dt = 1
		self.time = arange(0, self.T, self.dt)
		self.t_rest = 0 

		self.V = zeros(len(self.time)) - 65 # resting potential is -65
		self.R = 1 # resitances in kiloohms
		self.C = 10 # capacitance in microfarads
		self.tau = self.R * self.C # Time constant
		self.t_ref = 2 # refractory period
		self.threshold = -60 # spike threshold
		self.V_spike = 70 # spike delta
		self.spikes = zeros(len(self.time))

	def update(self, current=10):
		current = np.random.randint(10,20)
		print current
		for i,t in enumerate(self.time):
			if t > self.t_rest:
				self.V[i] = self.V[i-1] + (current*self.R)/self.tau*self.dt # Integration process
			if self.V[i] >= self.threshold: # spike and set new refractory period
				self.spikes[i] = 1
				self.V[i] += self.V_spike + self.V[i]
				self.t_rest = t + self.t_ref	

	def plot_neuron(self, index, n):
		ax = plt.subplot(2,n, index+1)
		plt.plot(self.time, self.get_firing_rates())
		ylabel('Membrane Potential (V)')

	def get_spikes(self):
		return self.spikes

	def get_voltages(self):
		return self.voltages

	def get_firing_rates(self):
		firing_rates = zeros(len(self.time))
		for i in range(len(self.time)):
			firing_rates[i] = np.sum(self.spikes[0:i])

		return firing_rates/self.T/10

neurons = []
for i in range(3):
	neurons.append(Neuron())
	neurons[i].update()


plt.figure(figsize=(20,4))
title('Integrate and Fire neuron on stimulation by a constant current')

for i in range(3):
	neurons[i].plot_neuron(i, 3)
xlabel('Time(msec)')

show()
