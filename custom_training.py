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

def create_model():
    input_layer = keras.Input(shape=(28,28), name="input")
    flatten = keras.layers.Flatten(name="flatten")(input_layer)
    mydense_1 = cl.MyDense(100,activation=tf.nn.relu, name="dense" )(flatten)
    output_layer = keras.layers.Dense(10, activation=tf.nn.softmax, name="output")(mydense_1)
    model = keras.models.Model(inputs=input_layer,outputs=output_layer)
    return model

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
              
def custom_training(train_images,train_labels):
    
    model = create_model()
    
    n_epochs = 5
    batch_size = 32
    n_steps = len(train_images) // batch_size
    optimizer = keras.optimizers.Nadam(lr=0.01)
    loss_fn = keras.losses.mean_squared_error
    mean_loss = keras.metrics.Mean()
    
    for epoch in range(1, n_epochs+1):
        print("Epoch {}/{}".format(epoch,n_epochs))
        for step in range(1, n_steps+1):
            X_batch, y_batch = random_batch(train_images,train_labels)
            y_batch = tf.one_hot(y_batch,10)
            with tf.GradientTape() as tape:
                y_pred = model(X_batch, training=True)
                main_loss = tf.reduce_mean(loss_fn(y_batch,y_pred))
                loss = tf.add_n([main_loss]+ model.losses)
            gradients = tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(gradients,model.trainable_variables))
            mean_loss(loss)
            on_train_batch_end(model,X_batch)
            #print_status(step*batch_size,len(train_labels),mean_loss)
        print_status(len(train_labels),len(train_labels),mean_loss)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    





