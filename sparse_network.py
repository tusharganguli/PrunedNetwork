#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 18:03:11 2021

@author: tushar
"""
import numpy as np

class SparseNetwork:
    def __init__(self, layer, activation_result, update_freq="step"):
        self.layer = layer
        self.activation_result = activation_result
        self.update_freq = update_freq
        
    def update_frequency(self):
        weights = self.layer.get_weights()
        activation_array = np.array(self.activation_result)
        activation_dim = activation_array.shape
        for step in range(activation_dim[1]):
            weights[2] = np.where(activation_array[0][step]>0,weights[2]+1,weights[2])
        self.layer.set_weights(weights)
        
    def sparsify(self):
        pass