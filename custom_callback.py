#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 21:37:06 2021

@author: tushar
"""

from tensorflow import keras

import sparse_network as sn

class MyCallback(keras.callbacks.Callback):
    def __init__(self, data_set, pruning_type, pruning_pct, 
                 pruning_chg, sparse_update_freq):
        """ Save params in constructor
        """
        if pruning_type != "neurons" and pruning_type != "weights":
            raise ValueError("Invalid pruning type supplied in callback")
            
        self.pruning_type = pruning_type
        self.pruning_pct = pruning_pct
        self.pruning_chg = pruning_chg
        self.sparse_update_freq = sparse_update_freq
        
        self.sn = sn.SparseNetwork()
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1)%self.sparse_update_freq != 0 or epoch == 0:
            return
        if self.pruning_type == "neurons":
            self.sn.sparsify_neurons(self.model,self.pruning_pct)
        elif self.pruning_type == "weights":
            self.sn.sparsify_weights(self.model,self.pruning_pct)
        
        self.pruning_pct += self.pruning_chg


        
        
        
        
       
               