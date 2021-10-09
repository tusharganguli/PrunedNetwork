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
import custom_model as cmod

class ModelRun():
    
    def __init__(self,data_set, log_dir):
        # Load dataset
        
        self.data_set = data_set
        data_obj = data.Data(data_set)
        (self.valid_img,self.train_img,self.valid_labels,
         self.train_labels,self.test_images,self.test_labels) = data_obj.load_data()
        
        self.df = self.__create_data_frame()
        
        self.log_dir = log_dir + "/" 
        self.log_dir = os.path.join(os.curdir,self.log_dir)
        
        # hyperparameters
        self.optimizer = "sgd"
        self.loss = "sparse_categorical_crossentropy"
        self.metrics = "accuracy"
        
    def evaluate(self, run_type, epochs, num_layers, 
                  neuron_cnt, num_runs, pruning_type, 
                  training_accuracy = 0.8,
                  pruning_pct=0, pruning_change=0,
                  prune_accuracy_threshold=.5,
                  prune_freq=1 ):
        
        history_list = []
        evaluate_list = []
        
        for runs in range(num_runs):   
            tf.keras.backend.clear_session()
            model = self.__create_model(run_type,neuron_cnt, num_layers)
            log_file_name = run_type + "_epoch_" + str(epochs) + \
                            "_num_layers_" + str(num_layers)
            
            cb = cc.PruningCallback(self.data_set, run_type, pruning_type,
                                    training_accuracy,
                                    pruning_pct, pruning_change,
                                    prune_accuracy_threshold,
                                    prune_freq)
            
            if run_type == "standard":
                run_log_dir = self.__get_run_logdir(self.log_dir,log_file_name)
                tensorboard_cb = keras.callbacks.TensorBoard(run_log_dir)
                model.compile(optimizer=self.optimizer, loss=self.loss, 
                              metrics=[self.metrics])
                history = model.fit(self.train_img, self.train_labels, 
                                    epochs=epochs,
                                    validation_data=(self.valid_img,self.valid_labels),
                                    callbacks=[cb,tensorboard_cb])
            elif run_type == "sparse":
                log_file_name += "_pruning_pct_" + str(pruning_pct)
                if pruning_change > 0:
                    log_file_name += "_increasing"
                elif pruning_change < 0:
                    log_file_name += "_decreasing"    
                log_file_name += "_" + pruning_type + "_pruning_" + str(pruning_change)
                log_file_name += "_prune_accuracy_threshold_" + str(prune_accuracy_threshold)
                
                run_log_dir = self.__get_run_logdir(self.log_dir,log_file_name)
                tensorboard_cb = keras.callbacks.TensorBoard(run_log_dir)
                model.compile(optimizer=self.optimizer, 
                              loss=self.loss,
                              metrics=[self.metrics], 
                              run_eagerly=True)
    
                history = model.fit(self.train_img, 
                                    self.train_labels, 
                                    epochs=epochs,
                                    validation_data=(self.valid_img,self.valid_labels),
                                    callbacks=[cb,tensorboard_cb])
            
            history_list.append(history)    
            eval_result = model.evaluate(self.test_images,self.test_labels)
            evaluate_list.append(eval_result)
        (train_loss,train_accuracy, 
         val_loss, val_accuracy,
         test_loss, test_accuracy) = self.__generate_avg(history_list, 
                                                         evaluate_list)
        (total_trainable_wts,
         total_sparsed_wts,sparse_pct) = self.__generate_model_summary(model)
        sparse_pct = sparse_pct.numpy()
            
        data = [run_type, pruning_type, epochs,
                num_layers, pruning_pct,
                prune_accuracy_threshold,
                sparse_pct,
                train_accuracy, val_accuracy,
                test_accuracy, train_loss,
                val_loss, test_loss ]
        df2 = pd.DataFrame([data], columns=list(self.df))
        self.df = self.df.append(df2,ignore_index = True)
        del model
        
    def write_to_file(self, filename):
        # create excel writer object
        writer = pd.ExcelWriter(filename)
        # write dataframe to excel
        self.df.to_excel(writer)
        # save the excel
        writer.save()
        self.df = self.df.iloc[0:0]
        
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
    
    def __generate_model_summary(self,model):
        total_trainable_wts = 0
        trainable_wts = model.trainable_weights
        for wts in trainable_wts:
            if "kernel" in wts.name:
                total_trainable_wts += tf.size(wts)
        non_trainable_vars = model.non_trainable_variables   
        total_sparsed_wts = 0
        for wts in non_trainable_vars:
            if "kernel_access" in wts.name:
                elements_equal_to_value = tf.equal(wts, 1)
                as_ints = tf.cast(elements_equal_to_value, tf.int32)
                count = tf.reduce_sum(as_ints)
                total_sparsed_wts += count        
        tf.print("Trainable variables:",total_trainable_wts)
        tf.print("Variables pruned:",total_sparsed_wts)
        sparse_pct = (total_sparsed_wts/total_trainable_wts)*100
        tf.print("Sparse percentage:",sparse_pct)
        return (total_trainable_wts,total_sparsed_wts,sparse_pct)
        
    def __create_data_frame(self):
        
        df = pd.DataFrame(columns = ['Model Type',
                                     'Pruning Type',
                                     'No Of Epochs', 
                                     'Intermediate Layers',
                                     'Pct Pruning', 
                                     'Pruning Accuracy Threshold',
                                     'Total Pruning(%) Achieved',
                                     'Training Accuracy',
                                     'Validation Accuracy',
                                     'Test Accuracy',
                                     'Training Loss',
                                     'Validation Loss',
                                     'Test Loss',
                                     ])

          
        return df
        
    def __get_run_logdir(self,log_dir,log_file_name):
        import time
        run_id = time.strftime("_%Y_%m_%d-%H_%M_%S")
        run_id = log_file_name+run_id
        return os.path.join(log_dir,run_id)
        
    def __create_model(self,run_type,neuron_cnt,num_layers):
        if num_layers > 4:
            raise ValueError("Maximum 4 layers supported")
            
        input_layer = keras.Input(shape=(28,28), name="input")
        flatten = keras.layers.Flatten(name="flatten")(input_layer)
        
        if run_type == "standard":    
            dense_1 = keras.layers.Dense(neuron_cnt[0],activation=tf.nn.relu, name="dense_1" )(flatten)
            final_dense = dense_1
            if num_layers >= 2:
                dense_2 = keras.layers.Dense(neuron_cnt[1],activation=tf.nn.relu, name="dense_2" )(dense_1)
                final_dense = dense_2
            if num_layers >= 3:
                dense_3 = keras.layers.Dense(neuron_cnt[2],activation=tf.nn.relu, name="dense_3" )(dense_2)
                final_dense = dense_3
            if num_layers >= 4:
                dense_4 = keras.layers.Dense(neuron_cnt[3],activation=tf.nn.relu, name="dense_4" )(dense_3)
                final_dense = dense_4
        
        if run_type == "sparse":    
            dense_1 = cl.MyDense(neuron_cnt[0],activation=tf.nn.relu, name="dense_1" )(flatten)
            final_dense = dense_1
            if num_layers >= 2:
                dense_2 = cl.MyDense(neuron_cnt[1],activation=tf.nn.relu, name="dense_2" )(dense_1)
                final_dense = dense_2
            if num_layers >= 3:
                dense_3 = cl.MyDense(neuron_cnt[2],activation=tf.nn.relu, name="dense_3" )(dense_2)
                final_dense = dense_3
            if num_layers >= 4:
                dense_4 = cl.MyDense(neuron_cnt[3],activation=tf.nn.relu, name="dense_4" )(dense_3)
                final_dense = dense_4
        
        output_layer = keras.layers.Dense(10, activation=tf.nn.softmax, name="output")(final_dense)
        if run_type == "sparse":
            model = cmod.CustomModel(inputs=input_layer,outputs=output_layer)
        else:
            model = keras.models.Model(inputs=input_layer,outputs=output_layer)
        return model