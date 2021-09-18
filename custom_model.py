#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 21:27:25 2021

@author: tushar
"""

import tensorflow as tf
from tensorflow import keras

import custom_layer as cl
import sparse_network as sn


class CustomModel(keras.Model):
    def __init__(self, inputs=None):
        super(CustomModel, self).__init__()
        self.inputs = inputs
        self.flatten = tf.keras.layers.Flatten(name="flatten")
        self.custom_dense_1 = cl.MyDense(300, activation=tf.nn.relu)
        self.custom_dense_2 = cl.MyDense(100, activation=tf.nn.relu)
        self.output_layer = keras.layers.Dense(10, activation=tf.nn.softmax, name="output")
        self.sparsify = tf.Variable(0,trainable=False, dtype=tf.int32)
    
    def call(self, inputs):
        flatten_data = self.flatten(inputs)
        dense_1_out = self.custom_dense_1(flatten_data)
        #self.__update_frequency(self.custom_dense_1,dense_1_out)
        dense_2_out = self.custom_dense_2(dense_1_out)
        #self.__update_frequency(self.custom_dense_2,dense_2_out)
        return self.output_layer(dense_2_out)
    
    def __update_frequency( self, layer, activation_data):
            weights = layer.get_weights()
            activation_dim = activation_data.shape
            for step in range(0,activation_dim[0]-1):
                weights[2] = tf.where(activation_data[step]>0,
                                                 weights[2]+1, weights[2])
            layer.set_weights(weights)
    
    """
    def train_step(self, data):
        #tf.print("enter CustomModel.call")
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # update the neuron frequency of all layers
        self.__on_train_end(x)
        
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    def __on_train_end(self, data):
        # store the data if possible
        pass
        

    
    def __update_gradients(self, gradients):
        idx = 0
        for layer in self.layers:
            if not layer.trainable_variables:
                continue
            if isinstance(layer,cl.MyDense) == True:
                weights = layer.get_weights()
                sorted_idx = weights[3]
                slice_len = weights[4]
                sliced_idx = sorted_idx[:slice_len]
                # remove the gradient value for all the selected neurons
                gradients[idx][:][sliced_idx] = 0
                gradients[idx+1][sliced_idx] = 0
            idx += 1    
                
        return gradients
    
def frequency_activation(self,X):
        kernel_dim = self.kernel.shape
        zero_wts = tf.zeros(kernel_dim[1])
        for step in range(kernel_dim[0]):
            self.sparsified_kernel[step].assign(tf.where(self.neuron_freq>0,self.kernel[step],zero_wts))
        self.xxxx = self.activation(tf.matmul(X,self.sparsified_kernel) + self.bias)
        return self.xxxx
    

"""
