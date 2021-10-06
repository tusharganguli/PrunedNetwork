#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 06:54:34 2021

@author: tushar
"""

import tensorflow as tf
from tensorflow import keras

class MyDense(keras.layers.Layer):
    def __init__(self,units,activation=None, **kwargs):
        super(MyDense,self).__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        
        
        
    def build(self, input_shape):
        # initialize the weights in this layer
        self.kernel = self.add_weight(name="kernel",shape=[input_shape[-1],self.units],
                                      initializer="glorot_uniform", trainable=True)
        #initialize the bias
        self.bias = self.add_weight(name="bias",shape=[self.units], 
                                    initializer="zeros", trainable=True)
        
        self.neuron_freq = self.add_weight(name="neuron_freq", shape=[self.units], 
                                           initializer="zeros", dtype=tf.int32,
                                           trainable=False)
        self.kernel_access = self.add_weight(name="kernel_access",
                                             shape=[input_shape[-1],self.units],
                                             initializer="zeros",
                                             dtype=tf.int32,
                                             trainable=False)
        super().build(input_shape)
        
        
    def call(self, X):
        activation_result = self.activation(tf.matmul(X,self.kernel) + self.bias)
        return activation_result
    
    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape.as_list()[:-1]+ [self.units])
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units":self.units, 
                "activation": keras.activations.serialize(self.activation)}
