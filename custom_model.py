#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 21:27:25 2021

@author: tushar
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

import custom_layer as cl
import sparse_network as sn

def update_frequency(layer, activation_data):
        weights = layer.get_weights()
        activation_dim = activation_data.shape
        for step in range(0,activation_dim[0]-1):
            weights[2] = tf.where(activation_data[step]>0,
                                             weights[2]+1, weights[2])
        layer.set_weights(weights)


class CustomModel(keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.input_layer = tf.keras.Input(shape=(28,28), name="input")
        self.flatten = tf.keras.layers.Flatten(name="flatten")
        self.custom_dense = cl.MyDense(100, activation=tf.nn.relu)
        self.output_layer = keras.layers.Dense(10, activation=tf.nn.softmax, name="output")
        
    
    def call(self, inputs):
        #tf.print("enter CustomModel.call")
        flatten_data = self.flatten(inputs)
        dense1_out = self.custom_dense(flatten_data)
        update_frequency(self.custom_dense,dense1_out)
        return self.output_layer(dense1_out)
"""
    def train_step(self, data):
        tf.print("enter CustomModel.call")
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
        #on_train_end(self,x)
        
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
     
def frequency_activation(self,X):
        kernel_dim = self.kernel.shape
        zero_wts = tf.zeros(kernel_dim[1])
        for step in range(kernel_dim[0]):
            self.sparsified_kernel[step].assign(tf.where(self.neuron_freq>0,self.kernel[step],zero_wts))
        self.xxxx = self.activation(tf.matmul(X,self.sparsified_kernel) + self.bias)
        return self.xxxx
    
def on_train_end(model,data):
    # input 
    inp = model.input
    layer_names = []
    outputs = []
    layer_obj = []
    for layer in model.layers:
        if isinstance(layer,cl.MyDense) == True:
            layer_obj += [layer]
            layer_names += [layer.name]
            outputs += [layer.output]
    # evaluation functions
    functors = [K.function([inp], [out]) for out in outputs]    
    
    # Testing
    count = 0
    for func in functors:
      tf.print("Layer Name: ",layer_names[count])
      activation_data = func(data) 
      update_frequency(layer_obj[count],activation_data)
      count += 1
"""
