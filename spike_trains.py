import numpy as np
import matplotlib.pyplot as plt

def read_generic_data(filename, x, y):
    D = np.zeros((x,y))
    with open(filename) as f:
        for line_num,line in enumerate(f):
            
            cols = line.strip().split(",")
            D[line_num,:] = map(float,cols)
    return standardize_data(D)

def standardize_data(D):
    [x,y] = D.shape
    for col in xrange(y):
        D[:,col] += abs(min(D[:,col]))

    for row in xrange(x):
        D[row,:] = D[row,:] * (100/np.mean(D[row,:])) # 100 is the mean, average firing rate per odor
        D[row,:] = map(int,D[row,:])
    
    return D

def generate_spike_trains(D, T, dt):
	M = T/dt
	spike_trains = np.zeros((len(D), len(D[0]), M))
	for i in range(M):
		X = np.random.uniform(size=M)
		for j in range(len(D)):
			for k in range(len(D[0])):
				if D[j,k] * dt/1000.0 > X[i]:
					spike_trains[j,k,i] = 1
				else:
					spike_trains[j,k,i] = 0
	return spike_trains

# Load in MNIST data
FEATURES = 784
NUM_EXAMPLES = 10000
D = read_generic_data('mnist10k.txt', NUM_EXAMPLES, FEATURES)
'''
X = np.zeros((NUM_EXAMPLES, 28, 28))
for i in range(NUM_EXAMPLES):
	X[i,:,:] = D[i,:].reshape(28,28).transpose()

# plot 10 examples images
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
'''

D = D[:100,:]
# Generate Spike Trains
T = 100
dt = 1
spike_trains = generate_spike_trains(D, T, dt)
'''
n=10
for i in range(10):
	# display original
    ax = plt.subplot(2, n+1, i + 1)
    plt.imshow(np.multiply(spike_trains[0,:,i].reshape(28,28).transpose(), D[0,:].reshape(28,28).transpose()))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
ax = plt.subplot(2, n+1, i + 1)
plt.imshow(D[0,:].reshape(28,28).transpose())
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()
'''
'''
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
	plt.imshow(m.reshape(28,28).transpose() / 100.0, vmin=0, vmax=1)
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
plt.show()
'''
lineoffsets2 = 1
linelengths2 = 1
# Raster plots
nbins = T/dt
ntrials = 784
spikes = spike_trains[0,:,:]
print spikes.shape
fig = plt.figure()
plt.imshow(spikes, cmap='Greys',  interpolation='nearest')
   
plt.title('Example raster plot')
plt.xlabel('time')
plt.ylabel('trial')
plt.show()
