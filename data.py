#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 02:10:47 2021

@author: tushar
"""

class Data():
    def __init__(self,dataset,validation_split=0.1):
        self.dataset = dataset
        self.validation_split = validation_split
        
    def load_data(self):
        # Load MNIST dataset
        (train_images, train_labels), (test_images, test_labels) = self.dataset.load_data()
        validation_sz = train_images.shape[0] * self.validation_split
        # Normalize the input image so that each pixel value is between 0 and 1.
        # create validation data set
        valid_img, train_img = train_images[:validation_sz] / 255.0,\
                                train_images[validation_sz:] / 255.0
        valid_labels, train_labels = train_labels[:validation_sz],\
                                    train_labels[validation_sz:]
        test_images = test_images / 255.0
        return (valid_img,train_img,valid_labels,train_labels,test_images,test_labels)