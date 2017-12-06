import mnist_loader
import network
import numpy as np
from matplotlib import pyplot as plt
import cPickle as pickle

change_bottleneck_layer = False
change_small_worldness = True
# import data
training_data, validation_data, test_data,train_ae,val_ae,test_ae = mnist_loader.load_data_wrapper(make_binary =True)

# run nets
dat = []
disc = []
net = network.Network([784,30,10,30,784], cost=network.QuadraticCost,doHeavy = False,doSmallWorld=False)
net.large_weight_initializer()
net.SGD(train_ae, 5, 10, 0.5,evaluation_data=val_ae) # 10, 10, 0.5
dat.append([net.weights,net.biases])
disc += ['OG']
if change_bottleneck_layer:
    n_bottle = [2,5,10,20,30]
    for i in range(len(bottle)):
        # normal
        this_lay = n_bottle[i]
        net = network.Network([784,30,this_lay,30,784], cost=network.QuadraticCost,doHeavy = False,doSmallWorld=False)
        net.large_weight_initializer()
        net.SGD(train_ae, 5, 10, 0.5,evaluation_data=val_ae) # 10, 10, 0.5
        dat.append([net.weights,net.biases])
        disc += ['Bottle:' + str(this_lay)]
        # change random cnx
        net = network.Network([784,30,this_lay,30,784], cost=network.QuadraticCost,doHeavy = False,doSmallWorld=True,add_rand=10)
        net.large_weight_initializer()
        net.SGD(train_ae, 5, 10, 0.5,evaluation_data=val_ae) # 10, 10, 0.5
        dat.append([net.weights,net.biases])
        disc += ['Bottle (SW):' + str(this_lay)]
        pickle.dump((dat,disc), open( "bottle_weights.p", "wb" ))
    
if change_small_worldness:
    pct_try = [1,5,10,15,20]
    for i in range(len(pct_try)):
        net = network.Network([784,30,10,30,784], cost=network.QuadraticCost,doHeavy = False,doSmallWorld=True,add_rand = pct_try[i])
        net.large_weight_initializer()
        net.SGD(train_ae, 5, 10, 0.5,evaluation_data=val_ae) # 10, 10, 0.5
        dat.append([net.weights,net.biases])
        disc += ['Small World:' + str(pct_try)]
    pickle.dump((dat,disc), open( "small_world.p", "wb" ))
