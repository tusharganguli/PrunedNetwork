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

class PruningCallback(keras.callbacks.Callback):
    def __init__(self, data_set, run_type, pruning_type, training_accuracy,
                 pruning_pct, pruning_chg, 
                 prune_accuracy_threshold, prune_freq):
        """ Save params in constructor
        """
        self.run_type = run_type    
        self.pruning_type = pruning_type
        self.training_accuracy = training_accuracy
        self.pruning_pct = pruning_pct
        self.pruning_chg = pruning_chg
        self.prune_accuracy_threshold = prune_accuracy_threshold
        self.prune_freq = prune_freq
        self.accuracy = 0
        self.count = 0
        self.sn = sn.SparseNetwork()
        
    def on_epoch_begin(self, epoch, logs=None):
        if self.run_type == "standard":
            return
        
        if self.accuracy < self.prune_accuracy_threshold:
            return
        self.count += 1
        
        if self.count%self.prune_freq != 0:
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
        
        if self.accuracy >= self.training_accuracy:
            self.model.stop_training = 1
        
    
        
        
       
               