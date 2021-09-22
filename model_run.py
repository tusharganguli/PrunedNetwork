#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 19:16:49 2021

@author: tushar
"""

import os
import tensorflow as tf
from tensorflow import keras
# for writing to an excel file
import pandas as pd

import data
import custom_callback as cc
import custom_layer as cl
import custom_metrics as cm

class ModelRun():
    
    def __init__(self,data_set = keras.datasets.mnist):
        # Load dataset
        self.data_set = data_set
        data_obj = data.Data(data_set)
        (self.valid_img,self.train_img,self.valid_labels,
         self.train_labels,self.test_images,self.test_labels) = data_obj.load_data()
        
        self.df = self.__create_data_frame()
        
        self.log_dir = "my_logs/"#+str(num_layers)+ "/" + str(epochs) + "/" + str(run_type)
        self.log_dir = os.path.join(os.curdir,self.log_dir)
        
        # hyperparameters
        self.optimizer = "adam"
        self.loss = "sparse_categorical_crossentropy"
        self.metrics = "accuracy"
        
    def run_model(self, run_type, epochs, num_layers, 
                  num_runs, pruning_type, 
                  pruning_pct=0, pruning_change=0,
                  neuron_update_freq=1, sparse_update_freq=1 ):
        
        history_list = []
        evaluate_list = []
        
        for runs in range(num_runs):   
            tf.keras.backend.clear_session()
            model = self.__create_model(run_type,num_layers)
            log_file_name = run_type + "_epoch_" + str(epochs) + \
                            "_num_layers_" + str(num_layers)
            
            if run_type == "standard":
                run_log_dir = self.__get_run_logdir(self.log_dir,log_file_name)
                tensorboard_cb = keras.callbacks.TensorBoard(run_log_dir)
                model.compile(optimizer=self.optimizer, loss=self.loss, 
                              metrics=[self.metrics])
                history = model.fit(self.train_img, self.train_labels, 
                                    epochs=epochs,
                                    validation_data=(self.valid_img,self.valid_labels),
                                    callbacks=[tensorboard_cb])
            elif run_type == "sparse":
                if pruning_change > 0:
                    log_file_name = "increasing_" + log_file_name
                elif pruning_change < 0:
                    log_file_name = "decreasing_" + log_file_name    
                log_file_name = pruning_type + "_" + log_file_name
                log_file_name = log_file_name + "_pruning_pct_" + str(pruning_pct)
                run_log_dir = self.__get_run_logdir(self.log_dir,log_file_name)
                tensorboard_cb = keras.callbacks.TensorBoard(run_log_dir)
                custom_metrics = cm.CustomMetrics(name="batch_size")
                model.compile(optimizer=self.optimizer, 
                              loss=self.loss,
                              metrics=[self.metrics,custom_metrics]) 
                              #run_eagerly=True)
    
                sparse_cb = cc.MyCallback(self.data_set, pruning_type,
                                          pruning_pct, pruning_change, 
                                          neuron_update_freq,sparse_update_freq)
                history = model.fit(self.train_img, 
                                    self.train_labels, 
                                    epochs=epochs,
                                    validation_data=(self.valid_img,self.valid_labels),
                                    callbacks=[sparse_cb,tensorboard_cb])
            
            history_list.append(history)    
            eval_result = model.evaluate(self.test_images,self.test_labels)
            evaluate_list.append(eval_result)
        (train_loss,train_accuracy, 
         val_loss, val_accuracy,
         test_loss, test_accuracy) = self.__generate_avg(history_list, evaluate_list)
        data = [run_type, pruning_type, epochs,
                num_layers, pruning_pct,
                train_loss,train_accuracy, 
                val_loss, val_accuracy,
                test_loss, test_accuracy]
        df2 = pd.DataFrame([data], columns=list(self.df))
        self.df = self.df.append(df2,ignore_index = True)

    def write_to_file(self, filename):
        # create excel writer object
        writer = pd.ExcelWriter(filename)
        # write dataframe to excel
        self.df.to_excel(writer)
        # save the excel
        writer.save()
        
    def __generate_avg( self, history_list, evaluate_list):
        train_loss = train_accuracy = val_loss = 0
        val_accuracy = test_loss = test_accuracy = 0
        count = len(history_list)
        
        for idx in range(count):
            train_loss += history_list[idx].history["loss"][-1]
            train_accuracy += history_list[idx].history["accuracy"][-1]
            val_loss += history_list[idx].history["val_loss"][-1]
            val_accuracy += history_list[idx].history["val_accuracy"][-1]
            test_loss += evaluate_list[idx][0]
            test_accuracy += evaluate_list[idx][1]
        train_loss /= count
        train_accuracy /= count
        val_loss /= count
        val_accuracy /= count
        test_loss /= count
        test_accuracy /= count
        return (train_loss,train_accuracy, 
                val_loss, val_accuracy, 
                test_loss, test_accuracy )
     
    def __create_data_frame(self):
        
        df = pd.DataFrame(columns = ['Model Type',
                                     'Pruning Type',
                                     'No Of Epochs', 
                                     'Intermediate Layers',
                                     'Pct Pruning', 
                                     'Training Loss',
                                     'Training Accuracy',
                                     'Validation Loss',
                                     'Validation Accuracy',
                                     'Test Loss',
                                     'Test Accuracy',
                                     ])

          
        return df
        
    def __get_run_logdir(self,log_dir,log_file_name):
        import time
        run_id = time.strftime("_%Y_%m_%d-%H_%M_%S")
        run_id = log_file_name+run_id
        return os.path.join(log_dir,run_id)
        
    def __create_model(self,run_type,num_layers):
        if num_layers > 4:
            raise ValueError("Maximum 4 layers supported")
            
        input_layer = keras.Input(shape=(28,28), name="input")
        flatten = keras.layers.Flatten(name="flatten")(input_layer)
        l1_neurons = 300
        l2_neurons = 100
        l3_neurons = 100
        l4_neurons = 50
        
        if run_type == "standard":    
            dense_1 = keras.layers.Dense(l1_neurons,activation=tf.nn.relu, name="dense_1" )(flatten)
            final_dense = dense_1
            if num_layers >= 2:
                dense_2 = keras.layers.Dense(l2_neurons,activation=tf.nn.relu, name="dense_2" )(dense_1)
                final_dense = dense_2
            if num_layers >= 3:
                dense_3 = keras.layers.Dense(l3_neurons,activation=tf.nn.relu, name="dense_3" )(dense_2)
                final_dense = dense_3
            if num_layers >= 4:
                dense_4 = keras.layers.Dense(l4_neurons,activation=tf.nn.relu, name="dense_4" )(dense_3)
                final_dense = dense_4
        
        if run_type == "sparse":    
            dense_1 = cl.MyDense(l1_neurons,activation=tf.nn.relu, name="dense_1" )(flatten)
            final_dense = dense_1
            if num_layers >= 2:
                dense_2 = cl.MyDense(l2_neurons,activation=tf.nn.relu, name="dense_2" )(dense_1)
                final_dense = dense_2
            if num_layers >= 3:
                dense_3 = cl.MyDense(l3_neurons,activation=tf.nn.relu, name="dense_3" )(dense_2)
                final_dense = dense_3
            if num_layers >= 4:
                dense_4 = cl.MyDense(l4_neurons,activation=tf.nn.relu, name="dense_4" )(dense_3)
                final_dense = dense_4
        
        output_layer = keras.layers.Dense(10, activation=tf.nn.softmax, name="output")(final_dense)
        model = keras.models.Model(inputs=input_layer,outputs=output_layer)
        return model