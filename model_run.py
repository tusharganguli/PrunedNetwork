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
import numpy as np

import data
import pruning_callback as pc
import custom_model as cmod

class ModelRun():
    
    def __init__(self,data_set, log_dir, prune_dir):
        # Load dataset
        
        self.data_set = data_set
        data_obj = data.Data(data_set)
        (self.valid_img,self.train_img,self.valid_labels,
        self.train_labels,self.test_images,self.test_labels) = data_obj.load_data()
        
        self.df = self.__create_data_frame()
        
        self.log_dir = log_dir + "/" 
        self.log_dir = os.path.join(os.curdir,self.log_dir)
        
        self.prune_dir = prune_dir + "/" 
        self.prune_dir = os.path.join(os.curdir,self.prune_dir)
        
        # hyperparameters
        self.optimizer = "sgd"
        self.loss = "sparse_categorical_crossentropy"
        self.metrics = "accuracy"
        
        # for creating the model
        self.epochs = 500
        self.num_layers = 3
        self.neuron_cnt = [300,100,50]
    
    def evaluate_standard(self,run_type, num_runs, final_training_accuracy):
        history_list = []
        evaluate_list = []
        epoch_list = []
        prune_pct_list = []
        
        for runs in range(num_runs):   
            tf.keras.backend.clear_session()
            model = self.__create_model(run_type)
            log_file_name = ""
            
            stop_cb = pc.StopCallback(final_training_accuracy)
            
            run_log_dir = self.__get_run_logdir(self.log_dir,log_file_name)
            tensorboard_cb = keras.callbacks.TensorBoard(run_log_dir)
            model.compile(optimizer=self.optimizer, loss=self.loss, 
                          metrics=[self.metrics])
            history = model.fit(self.train_img, self.train_labels, 
                                epochs=self.epochs,
                                validation_data=(self.valid_img,self.valid_labels),
                                callbacks=[stop_cb,tensorboard_cb])
            
            num_epochs = stop_cb.get_num_epochs()
            epoch_list.append(num_epochs)
            
            history_list.append(history)    
            eval_result = model.evaluate(self.test_images,self.test_labels)
            evaluate_list.append(eval_result)
            (total_trainable_wts,
             total_pruned_wts,prune_pct_achieved) = self.__generate_model_summary(model)
            prune_pct_achieved = prune_pct_achieved.numpy()
            prune_pct_list.append(prune_pct_achieved)
            del model
        self.__log_data(run_type, history_list, evaluate_list, 
                        epoch_list,prune_pct_list )
                
    def evaluate_pruning(self, run_type, num_runs, pruning_type, 
                  training_accuracy,
                  pruning_pct, pruning_change,
                  prune_at_accuracy,
                  prune_freq,
                  neuron_ctr_start_at_acc = 0, reset_neuron_count = False ):
        
        history_list = []
        evaluate_list = []
        epoch_list = []
        prune_pct_list = []
        
        log_file_name = "PrunePct_" + str(pruning_pct)
        if pruning_change > 0:
            log_file_name += "_Inc"
        elif pruning_change < 0:
            log_file_name += "_Dec"    
        log_file_name += "_PruneType_" + pruning_type # + "_pruning_" + str(pruning_change)
        log_file_name += "_NeuronUpdateAtAcc_" + str(neuron_ctr_start_at_acc)
        log_file_name += "_PruneAtAcc_" + str(prune_at_accuracy)
        log_file_name += "_FinalAcc_" + str(training_accuracy)
        if reset_neuron_count == True:
            log_file_name += "_ResetNeuron"    
        
        for runs in range(num_runs):   
            tf.keras.backend.clear_session()
            model = self.__create_model(run_type)
            
            stop_cb = pc.StopCallback(training_accuracy)
            
            run_log_dir = self.__get_run_logdir(self.log_dir,log_file_name)
            tensorboard_cb = keras.callbacks.TensorBoard(run_log_dir)
            pruning_cb = pc.PruningCallback(model,self.train_img, pruning_type,
                                            pruning_pct, pruning_change,
                                            prune_at_accuracy,
                                            prune_freq,
                                            reset_neuron_count,
                                            neuron_ctr_start_at_acc,
                                            self.prune_dir, log_file_name)
        
            model.compile(optimizer=self.optimizer, 
                          loss=self.loss,
                          metrics=[self.metrics], 
                          run_eagerly=True)

            history = model.fit(self.train_img, 
                                self.train_labels, 
                                epochs=self.epochs,
                                validation_data=(self.valid_img,self.valid_labels),
                                callbacks=[stop_cb,pruning_cb,tensorboard_cb])
        
            num_epochs = stop_cb.get_num_epochs()
            epoch_list.append(num_epochs)
            
            history_list.append(history)    
            eval_result = model.evaluate(self.test_images,self.test_labels)
            evaluate_list.append(eval_result)
            
            (total_trainable_wts,
             total_pruned_wts,prune_pct_achieved) = self.__generate_model_summary(model)
            prune_pct_achieved = prune_pct_achieved.numpy()
            prune_pct_list.append(prune_pct_achieved)
            del model
            
        self.__log_data(run_type, history_list, evaluate_list, epoch_list,
                        prune_pct_list )
    
    def evaluate_otp(self,run_type, num_runs, 
                     pruning_type, neuron_update_at_acc,
                     target_prune_pct, prune_at_accuracy,
                     final_training_accuracy):
        
        history_list = []
        evaluate_list = []
        epoch_list = []
        prune_pct_list = []
        
        log_file_name = run_type
        log_file_name += "_PruneType_" + pruning_type
        log_file_name += "_NeuronUpdateAtAcc_" + str(neuron_update_at_acc)
        log_file_name += "_PrunePct_" + str(target_prune_pct)
        log_file_name += "_FinalAcc_" + str(final_training_accuracy)
        
        for runs in range(num_runs):   
            tf.keras.backend.clear_session()
            model = self.__create_model(run_type)
            
            stop_cb = pc.StopCallback(final_training_accuracy)
            
            run_log_dir = self.__get_run_logdir(self.log_dir,log_file_name)
            tensorboard_cb = keras.callbacks.TensorBoard(run_log_dir)
            pruning_cb = pc.OTPCallback(model,pruning_type, neuron_update_at_acc,
                                        target_prune_pct, prune_at_accuracy,
                                        self.prune_dir, log_file_name)
        
            model.compile(optimizer=self.optimizer, 
                          loss=self.loss,
                          metrics=[self.metrics], 
                          run_eagerly=True)

            history = model.fit(self.train_img, 
                                self.train_labels, 
                                epochs=self.epochs,
                                validation_data=(self.valid_img,self.valid_labels),
                                callbacks=[stop_cb,pruning_cb,tensorboard_cb])
            
            num_epochs = stop_cb.get_num_epochs()
            epoch_list.append(num_epochs)
            
            history_list.append(history)    
            eval_result = model.evaluate(self.test_images,self.test_labels)
            evaluate_list.append(eval_result)
            (train_loss,train_accuracy, 
            val_loss, val_accuracy,
            test_loss, test_accuracy) = self.__generate_avg(history_list, 
                                                             evaluate_list)
            (total_trainable_wts,
            total_pruned_wts,prune_pct_achieved) = self.__generate_model_summary(model)
            prune_pct_achieved = prune_pct_achieved.numpy()
            prune_pct_list.append(prune_pct_achieved)
            del model
        self.__log_data(run_type, history_list, evaluate_list, epoch_list,
                        prune_pct_list )
        
    def evaluate_interval_pruning(self, run_type, num_runs, pruning_type, 
                                  final_training_accuracy,
                                  pruning_values,
                                  epoch_range,
                                  reset_neuron_count = False ):
        
        history_list = []
        evaluate_list = []
        epoch_list = []
        prune_pct_list = []
        
        target_pruning_pct = sum(pruning_values)
        log_file_name = run_type
        log_file_name += "_PrunePct_" + str(target_pruning_pct)
        log_file_name += "_PruneType_" + pruning_type
        log_file_name += "_FinalAcc_" + str(final_training_accuracy)
        if reset_neuron_count == True:
            log_file_name += "_ResetNeuron"    
            
        for runs in range(num_runs):   
            tf.keras.backend.clear_session()
            model = self.__create_model(run_type)
            
            stop_cb = pc.StopCallback(final_training_accuracy)
            
            run_log_dir = self.__get_run_logdir(self.log_dir,log_file_name)
            tensorboard_cb = keras.callbacks.TensorBoard(run_log_dir)
            pruning_cb = pc.IntervalPruningCallback(model,pruning_type,
                                                    pruning_values,
                                                    epoch_range,
                                                    reset_neuron_count,
                                                    self.prune_dir, log_file_name)
        
            model.compile(optimizer=self.optimizer, 
                          loss=self.loss,
                          metrics=[self.metrics], 
                          run_eagerly=True)

            history = model.fit(self.train_img, 
                                self.train_labels, 
                                epochs=self.epochs,
                                validation_data=(self.valid_img,self.valid_labels),
                                callbacks=[stop_cb,pruning_cb,tensorboard_cb])
            
            num_epochs = stop_cb.get_num_epochs()
            epoch_list.append(num_epochs)
            
            history_list.append(history)    
            eval_result = model.evaluate(self.test_images,self.test_labels)
            evaluate_list.append(eval_result)
            (train_loss,train_accuracy, 
             val_loss, val_accuracy,
             test_loss, test_accuracy) = self.__generate_avg(history_list, 
                                                         evaluate_list)
                                                                 
            (total_trainable_wts,
            total_pruned_wts,prune_pct_achieved) = self.__generate_model_summary(model)
            prune_pct_achieved = prune_pct_achieved.numpy()
            prune_pct_list.append(prune_pct_achieved)
            del model
        
        self.__log_data(run_type, history_list, evaluate_list, epoch_list,
                        prune_pct_list )

    def evaluate_optimal_pruning(self, run_type, num_runs, pruning_type,
                                 final_training_accuracy, 
                                 epoch_pruning_interval, num_pruning,  
                                 reset_neuron_count = False ):
        
        history_list = []
        evaluate_list = []
        epoch_list = []
        prune_pct_list = []
        
        log_file_name = run_type
        log_file_name += "_PruneType_" + pruning_type
        log_file_name += "_EpochInterval_" + str(epoch_pruning_interval)
        log_file_name += "_TotalPruning_" + str(num_pruning)
        log_file_name += "_FinalAcc_" + str(final_training_accuracy)
        
        if reset_neuron_count == True:
            log_file_name += "_ResetNeuron"    
            
        for runs in range(num_runs):   
            tf.keras.backend.clear_session()
            model = self.__create_model(run_type)
            
            stop_cb = pc.StopCallback(final_training_accuracy)
            
            run_log_dir = self.__get_run_logdir(self.log_dir,log_file_name)
            tensorboard_cb = keras.callbacks.TensorBoard(run_log_dir)
            pruning_cb = pc.OptimalPruningCallback(model,pruning_type,
                                                   epoch_pruning_interval,
                                                   num_pruning, 
                                                   reset_neuron_count,
                                                   self.log_dir,
                                                   self.prune_dir, log_file_name)
        
            model.compile(optimizer=self.optimizer, 
                          loss=self.loss,
                          metrics=[self.metrics], 
                          run_eagerly=True)

            history = model.fit(self.train_img, 
                                self.train_labels, 
                                epochs=self.epochs,
                                validation_data=(self.valid_img,self.valid_labels),
                                callbacks=[stop_cb,pruning_cb,tensorboard_cb])
            
            num_epochs = stop_cb.get_num_epochs()
            epoch_list.append(num_epochs)
            
            history_list.append(history)    
            eval_result = model.evaluate(self.test_images,self.test_labels)
            evaluate_list.append(eval_result)
            (train_loss,train_accuracy, 
             val_loss, val_accuracy,
             test_loss, test_accuracy) = self.__generate_avg(history_list, 
                                                         evaluate_list)
                                                                 
            (total_trainable_wts,
            total_pruned_wts,prune_pct_achieved) = self.__generate_model_summary(model)
            prune_pct_achieved = prune_pct_achieved.numpy()
            prune_pct_list.append(prune_pct_achieved)
            del model
        
        self.__log_data(run_type, history_list, evaluate_list, epoch_list,
                        prune_pct_list )

    def __log_data(self, pruning_type, history_list, evaluate_list, 
                   num_epoch_list, prune_pct_list, pruning_pct=0, prune_at_accuracy=0):
        
        (train_loss,train_accuracy, 
         val_loss, val_accuracy,
         test_loss, test_accuracy) = self.__generate_avg(history_list,
                                                         evaluate_list)
        epoch_avg = sum(num_epoch_list)/len(num_epoch_list)
        prune_achieved_avg = sum(prune_pct_list)/len(prune_pct_list)
        
        log_data = [pruning_type, epoch_avg,
                    pruning_pct,
                    prune_at_accuracy,
                    prune_achieved_avg,
                    train_accuracy, val_accuracy,
                    test_accuracy, train_loss,
                    val_loss, test_loss ]
        df2 = pd.DataFrame([log_data], columns=list(self.df))
        self.df = self.df.append(df2,ignore_index = True)

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
        trainable_wts_cnt = 0
        trainable_wts = model.trainable_weights
        pruned_wts = 0
        for wts in trainable_wts:
            if "kernel" in wts.name:
                trainable_wts_cnt = tf.add(trainable_wts_cnt,tf.size(wts))
                kernel_arr = wts.numpy()
                num_zeros = tf.size(kernel_arr[np.where(kernel_arr == 0)])
                pruned_wts = tf.add(pruned_wts,num_zeros)
                
        tf.print("Trainable variables:",trainable_wts_cnt)
        tf.print("Variables pruned:",pruned_wts)
        prune_pct = (pruned_wts/trainable_wts_cnt)*100
        tf.print("Prune percentage:",prune_pct)
        return (trainable_wts,pruned_wts,prune_pct)
    
    def __create_data_frame(self):
        
        df = pd.DataFrame(columns = ['Pruning Type',
                                     'No Of Epochs', 
                                     'Pct Pruning', 
                                     'Pruning At Accuracy',
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
        
    def __create_model(self,run_type):
        if self.num_layers > 4:
            raise ValueError("Maximum 4 layers supported")
            
        input_layer = keras.Input(shape=(28,28), name="input")
        flatten = keras.layers.Flatten(name="flatten")(input_layer)
        
        dense_1 = keras.layers.Dense(self.neuron_cnt[0],activation=tf.nn.relu, name="dense_1" )(flatten)
        final_dense = dense_1
        if self.num_layers >= 2:
            dense_2 = keras.layers.Dense(self.neuron_cnt[1],activation=tf.nn.relu, name="dense_2" )(dense_1)
            final_dense = dense_2
        if self.num_layers >= 3:
            dense_3 = keras.layers.Dense(self.neuron_cnt[2],activation=tf.nn.relu, name="dense_3" )(dense_2)
            final_dense = dense_3
        if self.num_layers >= 4:
            dense_4 = keras.layers.Dense(self.neuron_cnt[3],activation=tf.nn.relu, name="dense_4" )(dense_3)
            final_dense = dense_4
        
        output_layer = keras.layers.Dense(10, activation=tf.nn.softmax, name="output")(final_dense)
        if run_type != "standard":
            model = cmod.CustomModel(inputs=input_layer,outputs=output_layer)
        else:
            model = keras.models.Model(inputs=input_layer,outputs=output_layer)
        return model