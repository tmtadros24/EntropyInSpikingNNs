import mnist_loader
import network
import numpy as np
from matplotlib import pyplot as plt
import cPickle as pickle

# import data
training_data, validation_data, test_data,train_ae,val_ae,test_ae = mnist_loader.load_data_wrapper(make_binary =True)

# run nets
dat = []
net = network.Network([784,30,10,30,784], cost=network.QuadraticCost,doHeavy = False,doSmallWorld=True,add_rand=10)
net.large_weight_initializer()
net.SGD(train_ae, 5, 5, 0.5,evaluation_data=val_ae) # 10, 10, 0.5
dat.append([net.weights,net.biases]
           
net = network.Network([784,30,10,30,784], cost=network.QuadraticCost_withEnergy,
                            doHeavy = False,doSmallWorld=True,add_rand=10,lamb=0.01)
net.large_weight_initializer()
net.SGD(train_ae, 5, 5, 0.5,evaluation_data=val_ae)

dat.append([net.weights,net.biases]
pickle.dump(dat, open( "tmp_weights.p", "wb" ))
