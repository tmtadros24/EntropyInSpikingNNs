import mnist_loader
import network
import numpy as np
from matplotlib import pyplot as plt
import cPickle as pickle

change_bottleneck_layer = True
change_small_worldness = False
# import data
training_data, validation_data, test_data,train_ae,val_ae,test_ae = mnist_loader.load_data_wrapper(make_binary =True)

epochs = 10 # epochs
m_batch_size = 10 # mini batch size
# run nets
dat = []
disc = []
net = network.Network([784,30,10,30,784], cost=network.QuadraticCost,doHeavy = False,doSmallWorld=False)
net.large_weight_initializer()
net.SGD(train_ae, epochs, m_batch_size, 0.5,evaluation_data=val_ae) # 10, 10, 0.5
dat.append([net.weights,net.biases])
disc += ['OG']
if change_bottleneck_layer:
    #n_bottle = [2,5,10,20,30]
    n_bottle = np.arange(1,50,2) # higher density
    for i in range(len(bottle)):
        # normal
        this_lay = n_bottle[i]
        net = network.Network([784,30,this_lay,30,784], cost=network.QuadraticCost,doHeavy = False,doSmallWorld=False)
        net.large_weight_initializer()
        net.SGD(train_ae, epochs, m_batch_size, 0.5,evaluation_data=val_ae) # 10, 10, 0.5
        dat.append([net.weights,net.biases])
        disc += ['Bottle:' + str(this_lay)]
        # change random cnx
        net = network.Network([784,30,this_lay,30,784], cost=network.QuadraticCost,doHeavy = False,doSmallWorld=True,add_rand=10)
        net.large_weight_initializer()
        net.SGD(train_ae, epochs, m_batch_size, 0.5,evaluation_data=val_ae) # 10, 10, 0.5
        dat.append([net.weights,net.biases])
        disc += ['Bottle (SW):' + str(this_lay)]
        pickle.dump((dat,disc), open( "bottle_weights_more.p", "wb" ))
    
if change_small_worldness:
    #pct_try = [1,5,10,15,20]
    pct_try = np.arange(1,50,2) # higher density
    for i in range(len(pct_try)):
        net = network.Network([784,30,10,30,784], cost=network.QuadraticCost,doHeavy = False,doSmallWorld=True,add_rand = pct_try[i])
        net.large_weight_initializer()
        net.SGD(train_ae, epochs, m_batch_size, 0.5,evaluation_data=val_ae) # 10, 10, 0.5
        dat.append([net.weights,net.biases])
        disc += ['Small World:' + str(pct_try[i])]
    pickle.dump((dat,disc), open( "small_world_more.p", "wb" ))
