#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 18:03:11 2021

@author: tushar
"""
import tensorflow as tf

class SparseNetwork:
    def __init__(self, layer):
        self.layer = layer
    
        """
    def update_frequency(self):
        weights = self.layer.get_weights()
        activation_dim = self.activation_result.shape
        for step in range(activation_dim[1]):
            weights[2][step].assign(tf.where(self.activation_result[0][step]>0,weights[2][step]+1,weights[2][step]))
        self.layer.set_weights(weights)
        """
    def sparsify(self):
        pass
        