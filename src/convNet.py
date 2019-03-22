#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 23:04:42 2019

@author: user
"""
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
import matplotlib.image as mpimg
import DatasetLoad
#datasetName = 'CIFAR'
#train = 'yes'

class convNet:
    
    def defConvNet(self,dataset, inputs, activation, numNodes):
        
        if(dataset == 'CIFAR'):
            channels = 3
            classes = 10
        if(dataset == 'Fashion-MNIST'):
            channels = 1
            classes = 10
            
        if(activation == 'ReLu'):
            act = tf.nn.relu
        if(activation == 'tanh'):
            act = tf.nn.tanh
        if(activation == 'sigmoid'):
            act = tf.nn.sigmoid
        if(activation == 'swish'):
            act = tf.nn.swish
        
            
        if(len(numNodes)<1):
            print('Please provide atleast number of nodes value for atleast 5 layers')
            exit(0)
        
        
        
        
        
    
        if(dataset == 'CIFAR'):
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
        
        if(dataset =='Fashion-MNIST'):
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
                out_channel = out_channel*2
                inData = convNorm
        
        
            flatOut = tf.contrib.layers.flatten(convNorm)
            
            fullConneceted_1 = tf.contrib.layers.fully_connected(inputs=flatOut, num_outputs=128, activation_fn=act)
            
            fullConneceted_1 = tf.layers.batch_normalization(fullConneceted_1)
            
            fullConneceted_2 = tf.contrib.layers.fully_connected(inputs=fullConneceted_1, num_outputs=256, activation_fn=act)
            
            fullConneceted_2 = tf.layers.batch_normalization(fullConneceted_2)
            
            fullConneceted_3 = tf.contrib.layers.fully_connected(inputs=fullConneceted_2, num_outputs=512, activation_fn=act)
            
            fullConneceted_3 = tf.layers.batch_normalization(fullConneceted_3)    
            
            fullConneceted_4 = tf.contrib.layers.fully_connected(inputs=fullConneceted_3, num_outputs=1024, activation_fn=act)
            
            fullConneceted_4 = tf.layers.batch_normalization(fullConneceted_4)        
            
            
            outNN = tf.contrib.layers.fully_connected(inputs=fullConneceted_4, num_outputs=classes, activation_fn=None)
        
        return outNN