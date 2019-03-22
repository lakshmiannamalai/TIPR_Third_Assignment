#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:32:08 2019

@author: user
"""

from sklearn.preprocessing import StandardScaler
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

import convNet


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
        
    def testData(self,testfolder):
        NoImages = len(os.listdir(testfolder)) 
    
        inputs = np.zeros((NoImages,self.rows,self.cols,self.channel))
        itrNo = 0
        for imgName in os.listdir(testfolder):
              
            imgName = testfolder+ '/' + imgName
                
        
                
                
       
            img = mpimg.imread(imgName)
            print(img.shape)        
            inputs[itrNo,:] = img[:,:,0].reshape(self.rows,self.cols,1)
            itrNo += 1 
        return inputs

    def loadDataset(self,folder,batchID):
        if(self.dataset == 'CIFAR'):
            with open(folder + '/data_batch_' + str(batchID), mode='rb') as file:
            #with open(folder + '/test_batch', mode='rb') as file:
                data = cPickle.load(file)
            inputs = data['data']
            scaler = StandardScaler()
            scaler.fit(inputs)
            inputs = scaler.transform(inputs)
            
            self.images = inputs.reshape((len(data['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
            self.labels = data['labels']
        if(self.dataset == 'Fashion-MNIST'):
            fashion_mnist = input_data.read_data_sets(folder + '/', one_hot=True)
            self.images = np.zeros((len(fashion_mnist.train.images),28,28,1))
            for i in range(len(fashion_mnist.train.images)):
                self.images[i] = fashion_mnist.train.images[i].reshape(28,28,1)
            self.labels = fashion_mnist.train.labels
            print(self.labels[0])
            
    def loadTestDataset(self,folder):
        
        if(self.dataset == 'CIFAR'):
            #with open(folder + '/data_batch_' + str(batchID), mode='rb') as file:
            with open(folder + '/test_batch', mode='rb') as file:
                data = cPickle.load(file)
            inputs = data['data']
            scaler = StandardScaler()
            scaler.fit(inputs)
            inputs = scaler.transform(inputs)
            
            self.images = inputs.reshape((len(data['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
            self.labels = data['labels']
        if(self.dataset == 'Fashion-MNIST'):
            fashion_mnist = input_data.read_data_sets(folder + '/', one_hot=True)
            self.images = np.zeros((len(fashion_mnist.test.images),28,28,1))
            for i in range(len(fashion_mnist.test.images)):
                self.images[i] = fashion_mnist.test.images[i].reshape(28,28,1)
            self.labels = fashion_mnist.test.labels
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
    
    def normalize(self,imgOb):
        images = imgOb.images
        channel = imgOb.channel
        min_val = np.min(images)
        max_val = np.max(images)
        if(self.dataset == 'Fashion-MNIST'):
            images = (images-min_val) / (max_val-min_val)

        return images