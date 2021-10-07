#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 21:37:06 2021

@author: tushar
"""

import tensorflow as tf
from tensorflow import keras

import sparse_network as sn

class MyCallback(keras.callbacks.Callback):
    def __init__(self, data_set, pruning_type, pruning_pct, 
                 pruning_chg, start_prune_accuracy, sparse_update_freq):
        """ Save params in constructor
        """
        if pruning_type != "neurons" and pruning_type != "weights":
            raise ValueError("Invalid pruning type supplied in callback")
            
        self.pruning_type = pruning_type
        self.pruning_pct = pruning_pct
        self.pruning_chg = pruning_chg
        self.start_prune_accuracy = start_prune_accuracy
        self.sparse_update_freq = sparse_update_freq
        
        self.sn = sn.SparseNetwork()
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1)%self.sparse_update_freq != 0:
            return
        if self.sparse_update_freq != 1 and epoch == 0:
            return
        accuracy = logs["accuracy"]
        if accuracy < self.start_prune_accuracy:
            return
        
        if self.pruning_type == "neurons":
            self.sn.sparsify_neurons(self.model,self.pruning_pct)
        elif self.pruning_type == "weights":
            self.sn.sparsify_weights(self.model,self.pruning_pct)
        
        self.pruning_pct += self.pruning_chg
        self.__generate_model_summary()
        
    def __generate_model_summary(self):
        total_trainable_wts = 0
        trainable_wts = self.model.trainable_weights
        for wts in trainable_wts:
            if "kernel" in wts.name:
                total_trainable_wts += tf.size(wts)
        non_trainable_vars = self.model.non_trainable_variables   
        total_sparsed_wts = 0
        for wts in non_trainable_vars:
            if "kernel_access" in wts.name:
                elements_equal_to_value = tf.equal(wts, 1)
                as_ints = tf.cast(elements_equal_to_value, tf.int32)
                count = tf.reduce_sum(as_ints)
                total_sparsed_wts += count        
        tf.print("Trainable variables:",total_trainable_wts)
        tf.print("Variables pruned:",total_sparsed_wts)
        sparse_pct = (total_sparsed_wts/total_trainable_wts)*100
        tf.print("Sparse percentage:",sparse_pct)
        
       
               