#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 06:07:27 2021

@author: tushar
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import backend as K

import custom_layer as cl
import sparse_network as sn

def random_batch(X,y,batch_size=32):
    idx = np.random.randint(len(X),size=batch_size)
    return X[idx],y[idx]

def print_status(iteration, total, loss):
    print("{}\{} - {}".format(iteration,total,loss.result()))


def on_train_batch_end(model,data):
    #layer_names = [layer.name for layer in model.layers]
    
    # input 
    inp = model.input
    # all layer outputs            
    #outputs = [layer.output for layer in model.layers]
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
      #print('\n')
      print("Layer Name: ",layer_names[count])
      #print('\n')
      activation_data = func([data]) 
      #print(activation_data)
      sparse_network = sn.SparseNetwork(layer_obj[count],activation_data, update_freq="step")
      sparse_network.update_frequency()
      sparse_network.sparsify()
      count+=1

#@tf.function
def custom_training(model,train_images,train_labels):
    
    n_epochs = 5
    batch_size = 32
    n_steps = len(train_images) // batch_size
    optimizer = keras.optimizers.Nadam(lr=0.01)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mean_loss = keras.metrics.Mean()
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    
    for epoch in range(1, n_epochs+1):
        print("Epoch {}/{}".format(epoch,n_epochs))
        for step in range(1, n_steps+1):
            X_batch, y_batch = random_batch(train_images,train_labels)
            #y_batch = tf.one_hot(y_batch,10)
            with tf.GradientTape() as tape:
                y_pred = model(X_batch, training=True)
                main_loss = loss_fn(y_batch,y_pred)
                loss = tf.add_n([main_loss]+ model.losses)
            gradients = tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(gradients,model.trainable_variables))
            
            # Update training metric.
            train_acc_metric.update_state(y_batch, y_pred)
            mean_loss(loss)
            #on_train_batch_end(model,X_batch)
            
        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()
        print_status(len(train_labels),len(train_labels),mean_loss)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    





