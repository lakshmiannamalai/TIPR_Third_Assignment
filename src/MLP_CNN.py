#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 20:28:54 2019

@author: user
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.layers import Dense
from keras.models import Sequential
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import util
import DatasetLoad
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import convNet

datasetName = 'CIFAR'
folder = '../data/CIFAR-10'
if __name__ == '__main__':
#def MLPkeras(in_train,label_train,testData,testLabel,numNodes,activation):
    config = tf.ConfigProto(intra_op_parallelism_threads=0, 
                        inter_op_parallelism_threads=0, 
                        allow_soft_placement=True)
    
    imgOb = DatasetLoad.imgClass(datasetName)
    
    
    numLayers = 6
    numNodes = [imgOb.rows*imgOb.cols*3,10,imgOb.nOoFlABELS]
    activation = 'sigmoid'
    session = tf.Session(config=config)
    model = Sequential()
    model.add(Dense(numNodes[1], input_dim=numNodes[0], activation=activation,kernel_initializer='random_uniform',bias_initializer='zeros'))
    for i in range(2,len(numNodes)):
        model.add(Dense(numNodes[i], activation=activation,kernel_initializer='random_uniform',bias_initializer='zeros'))
        
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    if(datasetName == 'CIFAR'):
            n_batches = 1
    else:
            n_batches = 1
    in_test = []
    label_test = []
    epoch = 10
    batchSize = 128
    learning_rate = 0.001
    
    
    for batchID in range(1, n_batches + 1):
        imgOb.loadDataset(folder,batchID)
        images = imgOb.normalize(imgOb)
        if(datasetName == 'Fashion-MNIST'):
            images = images.reshape(len(images),imgOb.rows*imgOb.cols)
        if(datasetName == 'CIFAR'):
            images = images.reshape(len(images),imgOb.rows*imgOb.cols*3)
            
        labelID = imgOb.convertLabels()
        in_train, test, label_train, label = train_test_split(images, labelID, test_size=0.1)
        in_test.append(test)
        label_test.append(label)
        model.fit(in_train, label_train, epochs=epoch, batch_size=batchSize)
    
    
  
    predicted = model.predict_classes(in_test[0])
    scores = model.evaluate(in_test[0],label_test[0])
    
    accuracy = util.getAccuracy(np.argmax(label_test[0],axis=1),predicted)
    f1_macro,f1_micro = util.fi_macro_micro(np.argmax(label_test[0],axis=1),predicted, imgOb.nOoFlABELS)
    print(accuracy, f1_macro, f1_micro)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    
    tf.reset_default_graph()
    
    
    inputs = tf.placeholder(tf.float32, shape=(None, imgOb.rows, imgOb.cols, imgOb.channel), name='input_x')
    target =  tf.placeholder(tf.float32, shape=(None, imgOb.nOoFlABELS), name='output_y')
    
    numNodes = [64, 128, 256, 512, 128, 256, 512, 1024]
    convOb = convNet.convNet()
    model = convOb.defConvNet(datasetName, inputs, activation, numNodes)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=target))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    correct_pred = tf.equal(tf.argmax(model, 1), tf.argmax(target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
    
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=0, 
                            inter_op_parallelism_threads=1)) as session:
            # Initializing the variables
            session.run(tf.global_variables_initializer())
            # Training cycle
            #n_batches = 1
            in_test = []
            label_test = []
            for batchID in range(1, n_batches + 1):
                imgOb.loadDataset(folder,batchID)
                images = imgOb.normalize(imgOb)
                
                labelID = imgOb.convertLabels()
                in_train, test, label_train, label = train_test_split(images, labelID, test_size=0.1)
                in_test.append(test)
                label_test.append(label)
                for epoch in range(epoch):
                # Loop over all batches
                
                    for start in range(0, batchSize*2, batchSize):
                        end = min(start + batchSize, len(images))
                        #print(start,end)
                        #tf.train_neural_network(sess, optimizer, 1, images[start:end], labelID[start:end])
                        session.run(optimizer,feed_dict={inputs: in_train[start:end],target: label_train[start:end]})
                        loss = session.run(cost,feed_dict={inputs: in_train[start:end],target: label_train[start:end]})
                        valid_acc = session.run(accuracy,feed_dict={inputs: images[start:end],target: labelID[start:end]})
                        
                        print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(loss, valid_acc))
                    
                    #print_stats(sess, batch_features, batch_labels, cost, accuracy)
           
            prediction = session.run(model,feed_dict={inputs:in_test[0]})
            accuracy = util.getAccuracy(np.argmax(label_test,axis=1),np.argmax(prediction,axis=1))
            f1_macro,f1_micro = util.fi_macro_micro(np.argmax(label_test,axis=1),np.argmax(prediction,axis=1),10)
            print(accuracy,f1_macro, f1_micro)