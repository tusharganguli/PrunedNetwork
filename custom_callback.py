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

class MyCallback(keras.callbacks.Callback):
    def __init__(self, neuron_pct, pruning_stage):
        """ Save params in constructor
        """
        self.neuron_pct = neuron_pct
        self.pruning_stage = pruning_stage
        data_obj = data.Data(keras.datasets.mnist)
        (valid_img,train_img,valid_labels,
         train_labels,test_images,test_labels) = data_obj.load_data()
        self.train_img = train_img
        
        
    def on_epoch_end(self, epoch, logs=None):
        myList = []
        sorted_lst = []
        total_len = 0
        all_weights = []
        layer_cnt = 0
        neuron_len = []
        for layer in self.model.layers:
            if isinstance(layer,cl.MyDense) == True:
                weights = layer.get_weights()
                all_weights.append(weights)
                neuron_len.append(weights[2].shape[0])
                total_len += neuron_len[-1]
                myList.extend([(layer_cnt,layer,i,weights[2][i]) for i in range(neuron_len[-1])])
                layer_cnt += 1
                #tf.print(myList)
        sorted_lst = sorted(myList,key=lambda x: (x[3]))
        bottom_n = tf.cast(total_len * (self.neuron_pct/100),dtype=tf.int32)
        sorted_arr = np.array(sorted_lst)
        sorted_arr[0:bottom_n,3] = 0
        sorted_lst = sorted(sorted_arr,key=lambda x: (x[0],x[2]))
        sorted_arr = np.array(sorted_lst)
        start = 0
        for idx in range(layer_cnt):
            end = start+neuron_len[idx]
            all_weights[idx][2] = sorted_arr[start:end,3]
            sorted_lst[start][1].set_weights(all_weights[idx])
            start = end
                
    
    def on_train_batch_end( self, batch, logs=None):
        
        batch_sz = logs.get("batch_size")
        
        if batch == 0:
            self.__create_functors()
        
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
       
               