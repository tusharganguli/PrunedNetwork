#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 19:16:49 2021

@author: tushar
"""

import os
import tensorflow as tf
from tensorflow import keras

import data
import custom_callback as cc
import custom_layer as cl
import custom_metrics as cm

class ModelRun():
    
    def __init__(self,data_set = keras.datasets.mnist):
        # Load MNIST dataset
        data_obj = data.Data(data_set)
        (self.valid_img,self.train_img,self.valid_labels,
         self.train_labels,self.test_images,self.test_labels) = data_obj.load_data()


    def run_model(self,run_type,epochs,num_layers, num_runs, pruning_pct=20, pruning_stage=500):
        log_dir = "my_logs/"+str(num_layers)+ "/" + str(epochs) + "/" + str(run_type)
        log_dir = os.path.join(os.curdir,log_dir)
        
        for runs in range(num_runs):   
            tf.keras.backend.clear_session()
            model = self.__create_model(run_type,num_layers)
            log_file_name = run_type + "_epoch_" + str(epochs) + \
                            "_num_layers_" + str(num_layers)
            
            if run_type == "standard":
                run_log_dir = self.__get_run_logdir(log_dir,log_file_name)
                tensorboard_cb = keras.callbacks.TensorBoard(run_log_dir)
                model.compile(optimizer="adam", 
                              loss="sparse_categorical_crossentropy", 
                              metrics=["accuracy"], 
                              run_eagerly=True)
                history = model.fit(self.train_img, self.train_labels, 
                                    epochs=epochs,
                                    validation_data=(self.valid_img,self.valid_labels),
                                    callbacks=[tensorboard_cb])
            elif run_type == "sparse":
                log_file_name = log_file_name + "_pruning_pct_" + str(pruning_pct)
                run_log_dir = self.__get_run_logdir(log_dir,log_file_name)
                tensorboard_cb = keras.callbacks.TensorBoard(run_log_dir)
                custom_metrics = cm.CustomMetrics(name="batch_size")
                model.compile(optimizer="adam", 
                              loss="sparse_categorical_crossentropy",
                              metrics=["accuracy",custom_metrics], 
                              run_eagerly=True)
    
                cb = cc.MyCallback(pruning_pct, pruning_stage)
                history = model.fit(self.train_img, self.train_labels, 
                                      epochs=epochs,
                                      validation_data=(self.valid_img,self.valid_labels),
                                      callbacks=[cb,tensorboard_cb])
                
            model.evaluate(self.test_images,self.test_labels)

    def __get_run_logdir(self,log_dir,log_file_name):
        import time
        run_id = time.strftime("_%Y_%m_%d-%H_%M_%S")
        run_id = log_file_name+run_id
        return os.path.join(log_dir,run_id)
        
    def __create_model(self,run_type,num_layers):
        if num_layers > 3:
            raise ValueError("Maximum 3 layers supported")
            
        input_layer = keras.Input(shape=(28,28), name="input")
        flatten = keras.layers.Flatten(name="flatten")(input_layer)
        l1_neurons = 300
        l2_neurons = 100
        l3_neurons = 100
        
        if run_type == "standard":    
            dense_1 = keras.layers.Dense(l1_neurons,activation=tf.nn.relu, name="dense_1" )(flatten)
            final_dense = dense_1
            if num_layers >= 2:
                dense_2 = keras.layers.Dense(l2_neurons,activation=tf.nn.relu, name="dense_2" )(dense_1)
                final_dense = dense_2
            if num_layers >= 3:
                dense_3 = keras.layers.Dense(l3_neurons,activation=tf.nn.relu, name="dense_3" )(dense_2)
                final_dense = dense_3
        
        if run_type == "sparse":    
            dense_1 = cl.MyDense(l1_neurons,activation=tf.nn.relu, name="dense_1" )(flatten)
            final_dense = dense_1
            if num_layers >= 2:
                dense_2 = cl.MyDense(l2_neurons,activation=tf.nn.relu, name="dense_2" )(dense_1)
                final_dense = dense_2
            if num_layers >= 3:
                dense_3 = cl.MyDense(l3_neurons,activation=tf.nn.relu, name="dense_3" )(dense_2)
                final_dense = dense_3
        
        output_layer = keras.layers.Dense(10, activation=tf.nn.softmax, name="output")(final_dense)
        model = keras.models.Model(inputs=input_layer,outputs=output_layer)
        return model