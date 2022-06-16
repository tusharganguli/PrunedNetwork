#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 02:10:47 2021

@author: tushar
"""
import tensorflow as tf

class Data():
    def __init__(self,dataset,validation_split=0.1):
        self.dataset = dataset
        self.validation_split = validation_split
        
    def load_data(self):
        # Load MNIST dataset
        (train_img, train_labels), (test_img, test_labels) = self.dataset.load_data()
        validation_sz = tf.cast(train_img.shape[0] * self.validation_split, dtype=tf.int32)
        # Normalize the input image so that each pixel value is between 0 and 1.
        # create validation data set
        valid_img, train_img = train_img[:validation_sz] / 255.0,\
                                train_img[validation_sz:] / 255.0
        valid_labels, train_labels = train_labels[:validation_sz],\
                                    train_labels[validation_sz:]
        test_img = test_img / 255.0
        return (train_img,valid_img,test_img,train_labels,valid_labels,test_labels)