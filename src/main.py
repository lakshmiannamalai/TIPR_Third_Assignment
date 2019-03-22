#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 00:23:15 2019

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
import convNet
#datasetName = 'CIFAR'
#train = 'yes'

    




def getLayerConfig(configuration):
    s = configuration[1:-1]
    s1 = s.split(' ')
    
    numLayers = len(s1)#3
    numNodes = np.zeros(numLayers, dtype=int)
    for i in range(numLayers):
        numNodes[i] = int(s1[i])
    return numNodes


if __name__ == "__main__":
#    layerConf = [8 4 4]

    import argparse
    
    parser = argparse.ArgumentParser(description='MLP')
    parser.add_argument('--train', required=False,
                        help='path to train data')
    parser.add_argument('--test', required=False,
                        help='path to test data')
    parser.add_argument('--dataset', required=True,
                        help='dataset name: MNIST / Cat-Dog')
    parser.add_argument('--configuration', required=False,
                        help='[Nl_1 Nl_2 Nl_3]. Length of the vector is equal to number of layers')
    parser.add_argument('--activation', required=False,
                        help='Activation name: sigmoid/tanh/ReLu')
    
    args = parser.parse_args()
    
    if(args.train != None):
        if(args.train == None or args.dataset == None or args.configuration == None or args.activation == None):
            print('please provide 5 parameters')
            exit(0)
            
    if(args.train == None):
        if(args.test == None or args.dataset == None):
            print('please provide 2 parameters')
            exit(0)
            
    config = tf.ConfigProto(intra_op_parallelism_threads=0, 
                        inter_op_parallelism_threads=0, 
                        allow_soft_placement=True)
    
    datasetName = args.dataset
    if(args.train != None):
        numNodes = getLayerConfig(args.configuration)
    epochs = 10
    if(datasetName == 'CIFAR'):
        epochs = 50
    batch_size = 128
    learning_rate = 0.001
    if(args.train != None):
        activation = args.activation
    
        if(activation != 'sigmoid' and activation != 'ReLu' and activation != 'tanh' and activation != 'swish'):
            print('please provide a valid activation function (sigmoid / ReLu / tanh / swish)')
            exit(0)
    
    imgOb = DatasetLoad.imgClass(datasetName)
    save_model_path = '../Model/' + datasetName + '/'
    #save_model_path = '../Models/'
#    #numNodes = [64, 128, 256, 512, 128, 256, 512, 1024]
    if(args.train != None):
        tf.reset_default_graph()
    #    
    #    
        inputs = tf.placeholder(tf.float32, shape=(None, imgOb.rows, imgOb.cols, imgOb.channel), name='input_x')
        target =  tf.placeholder(tf.float32, shape=(None, imgOb.nOoFlABELS), name='output_y')
    #    
        #model = convNet.defConvNet(datasetName, inputs, activation, numNodes)
        convOb = convNet.convNet()
        model = convOb.defConvNet(datasetName, inputs, activation, numNodes)
    #    
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=target,name="prediction"))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    #    
        correct_pred = tf.equal(tf.argmax(model, 1), tf.argmax(target, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
    #    
        
        
        if(datasetName == 'CIFAR'):
                n_batches = 1
        else:
                n_batches = 1
    
        print('Training...')
        with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=0, 
                            inter_op_parallelism_threads=1)) as session:
            
            session.run(tf.global_variables_initializer())
    
            
            in_test = []
            label_test = []
            for batchID in range(1, n_batches + 1):
                imgOb.loadDataset(args.train,batchID)
                images = imgOb.normalize(imgOb)
                
                labelID = imgOb.convertLabels()
                in_train, test, label_train, label = train_test_split(images, labelID, test_size=0.1)
                in_test.append(test)
                label_test.append(label)
                for epoch in range(epochs):
                
                
                    for start in range(0, len(in_train), batch_size):
                        end = min(start + batch_size, len(in_train))
                        session.run(optimizer,feed_dict={inputs: in_train[start:end],target: label_train[start:end]})
                    loss = session.run(cost,feed_dict={inputs: in_train[0:1000],target: label_train[0:1000]})
                    valid_acc = session.run(accuracy,feed_dict={inputs: images[0:1000],target: labelID[0:1000]})
                        
                    print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(loss, valid_acc))
                    
                    
           
            prediction = session.run(model,feed_dict={inputs:in_test[0]})
            
            accuracy = util.getAccuracy(np.argmax(label_test[0],axis=1),np.argmax(prediction,axis=1))
            f1_macro,f1_micro = util.fi_macro_micro(np.argmax(label_test[0],axis=1),np.argmax(prediction,axis=1),imgOb.nOoFlABELS)
            #print(accuracy,f1_macro, f1_micro)
            print('Validation Accuracy :: %f' %accuracy)
            print('Validation Macro F1-Score :: %f' %f1_macro)
            print('Validation Micro F1-Score :: %f' %f1_micro)
            
            saver = tf.train.Saver()
            save_path = saver.save(session, save_model_path)
            
            
    
    if(args.train == None):
         testdata = args.test
         batchID = 1
         #images = imgOb.testData(testdata)
         #imgOb.loadTestDataset(args.test)
         imgOb.loadTestDataset(args.test)
         images = imgOb.normalize(imgOb)
         tf.reset_default_graph()
         saver = tf.train.import_meta_graph(save_model_path+datasetName+'.meta')
         label_test = imgOb.convertLabels()
         with tf.Session() as sess:
             
             saver.restore(sess, tf.train.latest_checkpoint(save_model_path))
             graph = tf.get_default_graph()
             for op in graph.get_operations():
                    print(op.name)
             op_to_restore = graph.get_tensor_by_name("fully_connected_4/BiasAdd:0")
             inputs = graph.get_tensor_by_name("input_x:0")
             prediction = sess.run(op_to_restore,feed_dict={inputs:images})
             accuracy = util.getAccuracy(np.argmax(label_test,axis=1),np.argmax(prediction,axis=1))
             f1_macro,f1_micro = util.fi_macro_micro(np.argmax(label_test,axis=1),np.argmax(prediction,axis=1),imgOb.nOoFlABELS)
             print('Test Accuracy :: %f' %accuracy)
             print('Test Macro F1-Score :: %f' %f1_macro)
             print('Test Micro F1-Score :: %f' %f1_micro)
        
             #print(accuracy,f1_macro, f1_micro)