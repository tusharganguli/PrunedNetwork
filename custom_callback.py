#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 21:37:06 2021

@author: tushar
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

import custom_layer as cl

class MyCallback(keras.callbacks.Callback):
    def __init__(self, neuron_pct):
        """ Save params in constructor
        """
        self.neuron_pct = neuron_pct
    
    def on_epoch_end(self, epoch,logs=None):
        for layer in self.model.layers:
            if isinstance(layer,cl.MyDense) == True:
                weights = layer.get_weights()
                bottom_n = tf.cast(weights[2].shape[0] * (1-(self.neuron_pct/100)),dtype=tf.int32)
                bottom_n_idx = np.argpartition(weights[2], bottom_n)
                weights[0][:][bottom_n_idx[:bottom_n]] = 0
                weights[2][:] = 0
                layer.set_weights(weights)
                