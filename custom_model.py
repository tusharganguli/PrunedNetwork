#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 21:27:25 2021

@author: tushar
"""

import tensorflow as tf
from tensorflow import keras

import sparse_network as sn


class CustomModel(keras.Model):
    def __init__(self, inputs,outputs):
        super(CustomModel, self).__init__(inputs,outputs)
        self.block_gradients = 0
        
    def train_step(self, data):
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
        
        if self.block_gradients == 1:
            gradients = self.__update_gradients(trainable_vars, 
                                                self.non_trainable_variables, 
                                                gradients)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    
    def __update_gradients(self, trainable_vars, non_trainable_vars, gradients):
        
        # retrieve all names of trainable vars
        trainable_name_lst = []
        for idx in range(len(trainable_vars)):
            trainable_name_lst.append(trainable_vars[idx].name)
            
        # retrieve all non trainable vars
        for idx in range(len(non_trainable_vars)):
            if "kernel_access" not in non_trainable_vars[idx].name:
                continue
            name = non_trainable_vars[idx].name.split('/')[0]
            matching = [s for s in trainable_name_lst if name in s and "kernel" in s]
            trainable_lst_idx = trainable_name_lst.index(matching[0])
            kernel_gradients = gradients[trainable_lst_idx].numpy()
            kernel_access = non_trainable_vars[idx]
            kernel_gradients[kernel_access == 1] = 0
            gradients[trainable_lst_idx] = kernel_gradients
        return gradients
        
    """
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
    """
