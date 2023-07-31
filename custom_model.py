#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 21:27:25 2021

@author: tushar
"""

import tensorflow as tf
from tensorflow import keras

class CustomModel(keras.Model):
    def __init__(self, inputs,outputs):
        super(CustomModel, self).__init__(inputs,outputs)
        self.neuron_update = False
        self.pruning_flag = False
            
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        
        # todo: replace this logic with setting the variables to non-trainable    
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # update the gradients based on pruning flag
        if self.pruning_flag == True:
            gradients = self.__update_gradients(gradients)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
            
    def __update_gradients(self, gradients):
        #print("__update_gradients")
        trainable_vars = self.trainable_variables
        for idx in range(len(trainable_vars)):
            if "kernel" not in trainable_vars[idx].name:
                continue
            kernel_gradients = gradients[idx].numpy()
            kernel_wts = trainable_vars[idx].numpy()
            kernel_gradients[kernel_wts == 0] = 0
            gradients[idx] = kernel_gradients
            
        return gradients
    
    def enable_pruning(self):
        self.pruning_flag = True
    
    