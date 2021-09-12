#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 21:37:06 2021

@author: tushar
"""

import tensorflow as tf
from tensorflow import keras

import custom_layer as cl

class MyCallback(keras.callbacks.Callback):
    def __init__(self, neuron_pct):
        """ Save params in constructor
        """
        self.neuron_pct = neuron_pct
    
    def on_train_batch_end(self, batch,logs=None):
        for layer in self.model.layers:
            if isinstance(layer,cl.MyDense) == True:
                weights = layer.get_weights()
                top_n = tf.cast(weights[2].shape[0] * (self.neuron_pct/100),dtype=tf.int32)
                top_n_idx = tf.math.top_k(weights[2], top_n, sorted=False)
                weights[0][:][top_n_idx.indices.numpy()] = 0
                layer.set_weights(weights)
                tf.print(weights)