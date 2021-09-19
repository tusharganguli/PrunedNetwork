#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 21:37:06 2021

@author: tushar
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import backend as K

import custom_layer as cl
import data
import sparse_network as sn

class MyCallback(keras.callbacks.Callback):
    def __init__(self, pruning_type, pruning_pct, pruning_stage):
        """ Save params in constructor
        """
        if pruning_type != "neurons" and pruning_type != "weights":
            raise ValueError("Invalid pruning type supplied in callback")
            
        self.pruning_type = pruning_type
        self.pruning_pct = pruning_pct
        self.pruning_stage = pruning_stage
        data_obj = data.Data(keras.datasets.mnist)
        (valid_img,train_img,valid_labels,
         train_labels,test_images,test_labels) = data_obj.load_data()
        self.train_img = train_img
        self.sn = sn.SparseNetwork()
        
    def on_epoch_end(self, epoch, logs=None):
        if self.pruning_type == "neurons":
            self.sn.sparsify_neurons(self.model,self.pruning_pct)
        elif self.pruning_type == "weights":
            self.sn.sparsify_weights(self.model,self.pruning_pct)
    
    def on_train_begin(self, logs=None):
        self.__create_functors()
            
    def on_train_batch_end( self, batch, logs=None):
        accuracy = logs.get("accuracy")
        if accuracy < 0.8:
            return
        
        batch_sz = logs.get("batch_size")
        
        if batch%self.pruning_stage == 0:
            np.random.shuffle(self.train_img)
            shuffle_data  = self.train_img[0:batch_sz]
            
            count = 0
            for func in self.functors:
                #tf.print("Layer Name: ",layer_names[count])
                activation_data = func(shuffle_data)
                self.__update_frequency(self.layer_obj[count],activation_data[0])
                count += 1
                
    def __create_functors(self):
        inp = self.model.input
        layer_names = []
        outputs = []
        self.layer_obj = []
        
        for layer in self.model.layers:
            if not isinstance(layer,cl.MyDense):
                continue
            self.layer_obj += [layer]
            layer_names += [layer.name]
            outputs += [layer.output]
        # evaluation functions
        self.functors = [K.function([inp], [out]) for out in outputs]    
        
        
    def __update_frequency( self, layer, activation_data):
        weights = layer.get_weights()
        activation_dim = activation_data.shape
        for step in range(0,activation_dim[0]-1):
            weights[2] = tf.where(activation_data[step]>0,
                                             weights[2]+1, weights[2])
        layer.set_weights(weights)
       
               