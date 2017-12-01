import mnist_loader
import network
import numpy as np

training_data, validation_data, test_data = mnist_loader.load_data()

num_training_examples = len(training_data[0])
num_val_examples = len(validation_data[0])
num_test_examples = len(test_data[0])
features = len(training_data[0][0])

Xtrain = np.zeros((num_training_examples, features))
Xtest = np.zeros((num_test_examples, features))
Xval = np.zeros((num_val_examples, features))

for i in range(num_training_examples):
	training_ex = training_data[0][i]
	Xtrain[i] = training_ex

for i in range(num_val_examples):
	val_ex = validation_data[0][i]
	Xval[i] = val_ex

for i in range(num_test_examples):
	test_ex = test_data[0][i]
	Xtest[i] = test_ex

training_data = zip(Xtrain, Xtrain)
validation_data = zip(Xval, Xval)
test_data = zip(Xtest, Xtest)

training_data = training_data[0:1]

net = network.Network([784, 500, 784], cost=network.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(Xtrain, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)
