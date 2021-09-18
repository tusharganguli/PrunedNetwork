#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 02:10:47 2021

@author: tushar
"""

from tensorflow import keras

class Data():
    def __init__(self,dataset,validation_sz=5000):
        self.dataset = dataset
        self.validation_sz = validation_sz
        
    def load_data(self):
        # Load MNIST dataset
        (train_images, train_labels), (test_images, test_labels) = self.dataset.load_data()
        
        # Normalize the input image so that each pixel value is between 0 and 1.
        # create validation data set
        valid_img, train_img = train_images[:self.validation_sz] / 255.0,\
                                train_images[self.validation_sz:] / 255.0
        valid_labels, train_labels = train_labels[:self.validation_sz],\
                                    train_labels[self.validation_sz:]
        test_images = test_images / 255.0
        return (valid_img,train_img,valid_labels,train_labels,test_images,test_labels)