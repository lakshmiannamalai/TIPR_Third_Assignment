#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 09:18:47 2019

@author: user
"""


import numpy as np


def getAccuracy(predicted, test):
    correct = 0
    for x in range(len(test)):
        
        if test[x] == predicted[x]:
            correct += 1
    return (correct/float(len(test))) * 100.0

def fi_macro_micro(predicted, testLabel, classNo):
    true_positives = np.zeros((classNo))
    false_positives = np.zeros((classNo))
    true_negatives = np.zeros((classNo))
    false_negatives = np.zeros((classNo))
    precision = np.zeros((classNo))
    recall = np.zeros((classNo))
 
    for i in range(0, len(testLabel)):
        for j in range(classNo-1):
            
            if testLabel[i] == j:
                if predicted[i] == testLabel[i]:
                    true_positives[j] += 1
                else:
                    false_negatives[j] += 1
            if testLabel[i] != j:
                if predicted[i] == j:
                    false_positives += 1
                else:
                    true_negatives += 1
 
    precision_macro = 0
    recall_macro = 0
    precision_micro_num = 0
    precision_micro_den = 0
    recall_micro_num = 0
    recall_micro_den = 0
    for j in range(classNo):
    
        if(true_positives[j] + false_positives[j]):
            precision[j] = true_positives[j] / (true_positives[j] + false_positives[j])
        precision_micro_num += true_positives[j]
        precision_micro_den += (true_positives[j] + false_positives[j])
        precision_macro += precision[j]
        
        if (true_positives[j] + false_negatives[j]):
            recall[j] = true_positives[j] / (true_positives[j] + false_negatives[j])
        recall_micro_num += true_positives[j]
        recall_micro_den += (true_positives[j] + false_negatives[j])
        recall_macro += recall[j]
        
    if(precision_macro+recall_macro):
        f1_macro = 2 * precision_macro * recall_macro / (precision_macro + recall_macro)
    else:
        f1_macro = 0
    if(precision_micro_den):
        precision_micro = precision_micro_num/precision_micro_den
    else:
        precision_micro = 0
    if(recall_micro_den):
        recall_micro = recall_micro_num/recall_micro_den
    else:
        recall_micro = 0
    if(precision_micro+recall_micro):
        f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro)
    else:
        f1_micro = 0
    return f1_macro/classNo,f1_micro

def rgb2gray(colorImg):

    rImg, gImg, bImg = colorImg[:,:,0], colorImg[:,:,1], colorImg[:,:,2]
    grayImg = 0.2989 * rImg + 0.5870 * gImg + 0.1140 * bImg

    return grayImg
#def networkKeras(inputs,labels):
    