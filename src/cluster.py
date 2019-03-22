#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 00:23:15 2019

@author: user
"""
import matplotlib
matplotlib.use('Agg')
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from sklearn.model_selection import train_test_split
import cPickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import ndimage
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import util as util
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

datasetName = 'CIFAR'
train = 'yes'
class imgClass:
    def __init__(self,dataset):
        self.dataset = dataset
        if(dataset == 'CIFAR'):
            self.labelName = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            self.channel = 3
            self.rows = 32
            self.cols = 32
            self.nOoFlABELS = 10
            
        if(dataset == 'Fashion-MNIST'):
            self.labelName = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal','Shirt','Sneaker','Bag','Ankle boot']
            #self.labelName = [‘T-shirt’, ‘Trouser’, ‘Pullover’, ‘Dress’, ‘Coat’, ‘Sandal’, ‘Shirt’, ‘Sneaker’, ‘Bag’,‘Ankle boot’]
            self.channel = 1
            self.rows = 28
            self.cols = 28
            self.nOoFlABELS = 10
        

#    def loadDataset(self,batchID):
#        if(self.dataset == 'CIFAR'):
#            with open('../data/CIFAR-10' + '/data_batch_' + str(batchID), mode='rb') as file:
#            
#                data = cPickle.load(file)
#        
#            self.images = data['data'].reshape((len(data['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
#            self.labels = data['labels']
#        if(self.dataset == 'Fashion-MNIST'):
#            fashion_mnist = input_data.read_data_sets('../data/Fashion-MNIST', one_hot=True)
#            self.images = np.zeros((len(fashion_mnist.train.images),28,28,1))
#            for i in range(len(fashion_mnist.train.images)):
#                self.images[i] = fashion_mnist.train.images[i].reshape(28,28,1)
#            self.labels = fashion_mnist.train.labels
#            print(self.labels[0])
        
    def loadDataset(self,batchID):
        if(self.dataset == 'CIFAR'):
            with open('../data/CIFAR-10' + '/data_batch_' + str(batchID), mode='rb') as file:
            
                data = cPickle.load(file)
            inputs = data['data']
            scaler = StandardScaler()
            scaler.fit(inputs)
            inputs = scaler.transform(inputs)
            #self.images = data['data'].reshape((len(data['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
            self.images = inputs.reshape((len(data['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
            self.labels = data['labels']
        if(self.dataset == 'Fashion-MNIST'):
            fashion_mnist = input_data.read_data_sets('../data/Fashion-MNIST' + '/', one_hot=True)
            self.images = np.zeros((len(fashion_mnist.train.images),28,28,1))
            for i in range(len(fashion_mnist.train.images)):
                self.images[i] = fashion_mnist.train.images[i].reshape(28,28,1)
            self.labels = fashion_mnist.train.labels
            print(self.labels[0])
        
    def convertLabels(self):
        if(self.dataset == 'CIFAR'):
            label = np.eye(self.nOoFlABELS)
            labelId = np.zeros((len(self.labels),self.nOoFlABELS))
            for i in range(len(self.labels)):
                labelId[i] = label[self.labels[i]]
                
        if(self.dataset == 'Fashion-MNIST'):
            labelId = self.labels
            
        return labelId
    
def defConvNet(dataset, inputs, activation, numNodes):
    
    if(dataset == 'CIFAR'):
        channels = 3
        classes = 10
    if(dataset == 'Fashion-MNIST'):
        channels = 1
        classes = 10
        
    if(activation == 'ReLu'):
        act = tf.nn.relu
    
        
    if(len(numNodes)<5):
        print('Please provide atleast number of nodes value for atleast 5 layers')
        exit(0)
    
    
    
    
    

    
    in_channel = channels
    out_channel = 64
    NoOfLayers = len(numNodes)
    inData = inputs
    for layer in range(NoOfLayers):
        convKernel = tf.Variable(tf.random_normal(shape=[numNodes[layer], numNodes[layer], in_channel, out_channel], mean=0, stddev=0.08))
        #convKernel = tf.Variable(tf.random_uniform(0.01,shape=[numNodes[layer], numNodes[layer], in_channel, out_channel]))
        
        convLayer = tf.nn.conv2d(inData, convKernel, strides=[1,1,1,1], padding='SAME')
        if(activation == 'ReLu'):
            convLayer = tf.nn.relu(convLayer)
        if(activation == 'sigmoid'):
            convLayer = tf.nn.sigmoid(convLayer)
        if(activation == 'tanh'):
            convLayer = tf.nn.tanh(convLayer)
        if(activation == 'swish'):
            convLayer = tf.nn.swish(convLayer)
        convPool = tf.nn.max_pool(convLayer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        convNorm = tf.layers.batch_normalization(convPool)
        in_channel = out_channel
        #out_channel = out_channel*2
        inData = convNorm


    flatOut = tf.contrib.layers.flatten(convNorm)
    
    fullConneceted_1 = tf.contrib.layers.fully_connected(inputs=flatOut, num_outputs=128, activation_fn=act)
    
    fullConneceted_1 = tf.layers.batch_normalization(fullConneceted_1)
    
    fullConneceted_2 = tf.contrib.layers.fully_connected(inputs=fullConneceted_1, num_outputs=64, activation_fn=act)
    
    fullConneceted_2 = tf.layers.batch_normalization(fullConneceted_2)
    
    fullConneceted_3 = tf.contrib.layers.fully_connected(inputs=fullConneceted_2, num_outputs=512, activation_fn=act)
    
    fullConneceted_3 = tf.layers.batch_normalization(fullConneceted_3)    
    
    fullConneceted_4 = tf.contrib.layers.fully_connected(inputs=fullConneceted_3, num_outputs=1024, activation_fn=act)
    
    fullConneceted_4 = tf.layers.batch_normalization(fullConneceted_4)        
    
    
    outNN = tf.contrib.layers.fully_connected(inputs=fullConneceted_2, num_outputs=classes, activation_fn=None)


    
    
    return outNN, fullConneceted_4

def getAccuracy(predicted,true,numClass):
    maxAcc = 0
    for i in range(numClass):
        s = 0
        for itr in range(len(predicted)):
            if(predicted[itr]==true[itr]):
                s += 1
        
        print(s)
        accuracy = s/float(len(predicted))
        if(accuracy > maxAcc):
            maxAcc = accuracy
        predicted = np.add(predicted,1)
        predicted = np.mod(predicted,numClass)
        print(accuracy)
    return maxAcc
        

#def normalize(imgOb):
#    images = imgOb.images
#    channel = imgOb.channel
#    min_val = np.min(images)
#    max_val = np.max(images)
#    images = (images-min_val) / (max_val-min_val)
#
#    return images
    
def normalize(imgOb):
    images = imgOb.images
    channel = imgOb.channel
    min_val = np.min(images)
    max_val = np.max(images)
    if(datasetName == 'Fashion-MNIST'):
        images = (images-min_val) / (max_val-min_val)
#    for i in range(channel):
#        mean = np.mean(images[:,:,:,i])
#        images[:,:,:,i] = images[:,:,:,i]-mean
#        var = (1.0/(imgOb.rows*imgOb.cols))*ndimage.variance(images[:,:,:,i])
#        images[:,:,:,i] = images[:,:,:,i]/np.sqrt(var)
#        print(mean, var)
    return images

if __name__ == "__main__":
#    layerConf = [8 4 4]
#    s = args.configuration[1:-1]
#    s1 = s.split(' ')
    
#    numLayers = len(s1)#3
#    numNodes = np.zeros(numLayers, dtype=int)
#    for i in range(numLayers):
#        numNodes[i] = int(s1[i])
    
    config = tf.ConfigProto(intra_op_parallelism_threads=0, 
                        inter_op_parallelism_threads=0, 
                        allow_soft_placement=True)
    
    
    
    epochs = 1
    batch_size = 128
    learning_rate = 0.001
    activation = 'sigmoid'
    
    imgOb = imgClass(datasetName)
    numNodes = [64, 128, 256, 512, 128, 256, 512, 1024]
    
    tf.reset_default_graph()
    
    
    inputs = tf.placeholder(tf.float32, shape=(None, imgOb.rows, imgOb.cols, imgOb.channel), name='input_x')
    target =  tf.placeholder(tf.float32, shape=(None, imgOb.nOoFlABELS), name='output_y')
    
    model, ftreLayer = defConvNet(datasetName, inputs, activation, numNodes)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=target))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    correct_pred = tf.equal(tf.argmax(model, 1), tf.argmax(target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
    
    
    save_model_path = '../Models/' + datasetName
    
    
    print('Training...')
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=0, 
                        inter_op_parallelism_threads=1)) as session:
        # Initializing the variables
        session.run(tf.global_variables_initializer())

        # Training cycle
        if(datasetName == 'CIFAR'):
            n_batches = 1
        else:
            n_batches = 1
            
        in_test = []
        label_test = []
        for batchID in range(1, n_batches + 1):
            imgOb.loadDataset(batchID)
            images = normalize(imgOb)
            
            labelID = imgOb.convertLabels()
            in_train, test, label_train, label = train_test_split(images, labelID, test_size=0.9)
            in_test.append(test)
            label_test.append(label)
            for epoch in range(epochs):
            # Loop over all batches
            
                for start in range(0, len(in_train), batch_size):
                    end = min(start + batch_size, len(images))
                    #print(start,end)
                    #tf.train_neural_network(sess, optimizer, 1, images[start:end], labelID[start:end])
                    session.run(optimizer,feed_dict={inputs: in_train[start:end],target: label_train[start:end]})
                    loss = session.run(cost,feed_dict={inputs: in_train[start:end],target: label_train[start:end]})
                    valid_acc = session.run(accuracy,feed_dict={inputs: images[start:end],target: labelID[start:end]})
                    
                    print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(loss, valid_acc))
                
                #print_stats(sess, batch_features, batch_labels, cost, accuracy)
       
        feature = session.run(ftreLayer,feed_dict={inputs:in_test[0]})
        #KM.KMeans(n_clusters=8, init=’k-means++’, n_init=10, max_iter=300, tol=0.0001, precompute_distances=’auto’, verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm=’auto’)
        kmeans = KMeans(n_clusters = 10, init='k-means++',n_init=10,max_iter=300,tol=0.0001,verbose=0).fit(feature)
        predictClass = kmeans.predict(feature)
        print(len(predictClass))
        accuracy = getAccuracy(predictClass,np.argmax(label_test[0],axis=1),imgOb.nOoFlABELS)
        labels = np.argmax(label_test[0],axis=1)
        print(len(labels))
        feature_embedded = TSNE(n_components=2).fit_transform(feature)
        #plt.plot(feature_embedded)
        plt.figure(figsize=(3, 3))
        
        
        plt.scatter(feature_embedded[:,0],feature_embedded[:,1],c = labels)
        plt.title('Cluster plot of feature')
        plt.xlabel('Feature x1')
        plt.ylabel('Feature x2')
        plt.savefig('myfig')  # saves the current figure into a pdf page
        plt.close()
        
        
    
    