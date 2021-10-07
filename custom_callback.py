#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 21:37:06 2021

@author: tushar
"""

import tensorflow as tf
from tensorflow import keras

import numpy as np

import sparse_network as sn

class MyCallback(keras.callbacks.Callback):
    def __init__(self, data_set, pruning_type, pruning_pct, 
                 pruning_chg, start_prune_at_accuracy, sparse_update_freq):
        """ Save params in constructor
        """
        if pruning_type != "neurons" and pruning_type != "weights":
            raise ValueError("Invalid pruning type supplied in callback")
            
        self.pruning_type = pruning_type
        self.pruning_pct = pruning_pct
        self.pruning_chg = pruning_chg
        self.start_prune_at_accuracy = start_prune_at_accuracy
        self.sparse_update_freq = sparse_update_freq
        self.accuracy = 0
        self.sn = sn.SparseNetwork()
        
    def on_epoch_begin(self, epoch, logs=None):
        if (epoch+1)%self.sparse_update_freq != 0:
            return
        if self.sparse_update_freq != 1 and epoch == 0:
            return
        if epoch == 0:
            return
        
        if self.accuracy < self.start_prune_at_accuracy:
            return
        
        wts = self.model.trainable_variables
        num_zeros = 0
        for w in wts:
            if "kernel" in w.name:
                kernel = w.numpy()
                num_zeros += kernel[np.where(kernel == 0)].size
        tf.print("zeros before sparsification:" + str(num_zeros))
        
        if self.pruning_type == "neurons":
            self.sn.sparsify_neurons(self.model,self.pruning_pct)
        elif self.pruning_type == "weights":
            self.sn.sparsify_weights(self.model,self.pruning_pct)
        
        wts = self.model.trainable_variables
        num_zeros = 0
        for w in wts:
            if "kernel" in w.name:
                kernel = w.numpy()
                num_zeros += kernel[np.where(kernel == 0)].size
        tf.print("zeros after sparsification:" + str(num_zeros))
        
        self.pruning_pct += self.pruning_chg
        
    def on_epoch_end(self,epoch,logs=None):
        self.accuracy = logs["accuracy"]
        
    
        
        
       
               