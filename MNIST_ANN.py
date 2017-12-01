import mnist_loader
import network
import numpy as np

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([784, 30, 10], cost=network.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)
