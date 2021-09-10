#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 06:54:34 2021

@author: tushar
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

class MyDense(keras.layers.Layer):
    def __init__(self,units,activation=None, **kwargs):
        super(MyDense,self).__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        print("self.units:"+ str(self.units))
        
    def build(self, input_shape):
        # initialize the weights in this layer
        self.kernel = self.add_weight(name="kernel",shape=[input_shape[-1],self.units],
                                      initializer="glorot_normal", trainable=True)
        #initialize the bias
        self.bias = self.add_weight(name="bias",shape=[self.units], initializer="zeros")
        self.neuron_freq = self.add_weight(name="neuron_freq", shape=[self.units], 
                                           initializer="zeros",trainable=False)
        super().build(input_shape)
    
    def standard_activation(self, X):
        return self.activation(tf.matmul(X,self.kernel) + self.bias)
    
    def frequency_activation(self,X):
        kernel_dim = self.kernel.shape
        initializer = tf.keras.initializers.Zeros()
        kernel = tf.Variable(lambda: initializer(shape=kernel_dim))
        zero_wts = tf.zeros(kernel_dim[1])
        tf.io.write_file("original.npy",tf.strings.as_string(self.kernel))
        for step in range(kernel_dim[0]):
            kernel[step].assign(tf.where(self.neuron_freq>0,self.kernel[step],zero_wts))
        tf.io.write_file("changed.npy",tf.strings.as_string(kernel))
        activation_result = self.activation(tf.matmul(X,self.kernel) + self.bias)
        return activation_result
        
    def call(self, X):
        total = tf.reduce_sum(self.neuron_freq)
        return tf.cond(total == 0, 
                       lambda: self.standard_activation(X),
                       lambda: self.frequency_activation(X))
        
    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape.as_list()[:-1]+ [self.units])
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units":self.units, 
                "activation": keras.activations.serialize(self.activation),
                "neuron_freq": self.neuron_freq}
