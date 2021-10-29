#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 21:37:06 2021

@author: tushar
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import os
from datetime import datetime
#import random

import prune_network as pn

class LogHandler:
    df = pd.DataFrame(columns = ['Epoch', 
                                 'Total Trainable Wts',
                                 'Total Pruned Wts',
                                 'Prune Percentage'
                                 ])
    
    def __init__(self, prune_dir, log_file_name):    
        self.prune_dir = prune_dir
        if not os.path.exists(self.prune_dir):
            os.makedirs(self.prune_dir)
        self.log_file_name = log_file_name
    
    def log(self, epoch, total_trainable_wts, total_pruned_wts, prune_pct):
        log_data = [epoch,total_trainable_wts.numpy(),
                    total_pruned_wts.numpy(),
                    prune_pct.numpy()]
        df2 = pd.DataFrame([log_data], columns=list(LogHandler.df))
        LogHandler.df = LogHandler.df.append(df2,ignore_index = True)
    
    def write_to_file(self):
        writer = pd.ExcelWriter(self.prune_dir + self.log_file_name + ".xls")
        LogHandler.df.to_excel(writer)
        # save the excel
        writer.save()
        LogHandler.df = LogHandler.df.iloc[0:0]
   
class PruningCallback(keras.callbacks.Callback):
    def __init__(self, model, train_data, pruning_type,
                 pruning_pct, pruning_chg, 
                 prune_at_accuracy, prune_freq,
                 reset_neuron_count,
                 neuron_start_at_acc,
                 prune_dir, file_name):
        """ Save params in constructor
        """
        self.train_data = train_data
        self.pruning_type = pruning_type
        self.pruning_pct = pruning_pct
        self.pruning_chg = pruning_chg
        self.prune_at_accuracy = prune_at_accuracy
        self.prune_freq = prune_freq
        self.reset_neuron_count =  reset_neuron_count
        self.accuracy = 0
        self.count = 0
        self.pn = pn.PruneNetwork(model)
        model.set_prune_network(self.pn)
        self.prev_loss = 1
        self.lh = LogHandler(prune_dir, file_name)
        
        self.neuron_start_at_acc = neuron_start_at_acc
        
    
    def on_epoch_begin(self, epoch, logs=None):
        if self.accuracy < self.prune_at_accuracy:
            return
        self.count += 1
        
        if self.count%self.prune_freq != 0:
            return
        
        num_zeros = self.pn.get_zeros()
        tf.print("zeros before pruning:" + str(num_zeros))
        
        if self.pruning_type == "neurons":
            self.pn.sparsify_neurons(self.model,self.pruning_pct)
        elif self.pruning_type == "weights":
            self.pn.sparsify_weights(self.model,self.pruning_pct)
        elif self.pruning_type == "absolute_weights":
            self.pn.sparsify_absolute_weights(self.model,self.pruning_pct)
        
        (total_trainable_wts,
         total_pruned_wts,prune_pct) = self.pn.get_prune_summary()
        tf.print("zeros after pruning:" + str(total_pruned_wts.numpy()))
        
        self.lh.log(epoch, total_trainable_wts, total_pruned_wts, prune_pct)
        
        self.pruning_pct += self.pruning_chg
        
        if self.reset_neuron_count == True:
            self.pn.reset_neuron_count()
            
    def on_train_end(self,logs=None):
        self.lh.write_to_file()
        
    def on_batch_end(self,batch,logs=None):
        self.accuracy = logs["accuracy"]
        # update the neuron frequency
        if self.accuracy >= self.neuron_start_at_acc:
            self.pn.enable_neuron_update()
    """
    def on_batch_end(self,batch,logs=None):
        self.accuracy = logs["accuracy"]
        # update the neuron frequency
        if self.accuracy >= self.neuron_ctr_start_at_acc:
            # generate a random batch of data of size 32
            np.random.shuffle(self.train_data)
            shuffle_data = self.train_data[0:32]
            #sample_idxs = np.unique(np.random.randint(self.train_data.shape[0], size=32))
            #shuffle_data  = self.train_data[sample_idxs]
            self.pn.update_neuron_frequency(shuffle_data)
    """        
        

class OTPCallback(keras.callbacks.Callback):
    def __init__(self, model,pruning_type, neuron_update_at_acc,
                 target_prune_pct, prune_at_accuracy,
                 prune_dir, log_file_name ):
        self.pruning_type = pruning_type
        self.neuron_update_at_acc = neuron_update_at_acc
        self.target_prune_pct = target_prune_pct
        self.prune_at_accuracy = prune_at_accuracy
        self.pruning_done = False
        self.pn = pn.PruneNetwork(model)
        model.set_prune_network(self.pn)
        self.lh = LogHandler(prune_dir, log_file_name)
        self.prune = "on_batch_end"
        
    def on_epoch_end(self, epoch, logs=None):
        if self.prune != "on_epoch_end":
            return
        self.__prune_and_log(epoch,logs["accuracy"])
        
        
    def on_batch_end(self,batch,logs=None):
        if self.pruning_done == True:
            return
        accuracy = logs["accuracy"]
        # update the neuron frequency
        if accuracy >= self.neuron_update_at_acc:
            self.pn.enable_neuron_update()
        
        if self.prune != "on_batch_end":
            return
        self.__prune_and_log(batch,accuracy)
        
    def on_train_end(self,logs=None):
        self.lh.write_to_file()
    
    def __prune_and_log(self, step, accuracy):
        if self.pruning_done == True:
            return
        
        if accuracy < self.prune_at_accuracy:
            return
        
        num_zeros = self.pn.get_zeros()
        tf.print("zeros before pruning:" + str(num_zeros))
        
        self.__prune()
        
        (total_trainable_wts,
         total_pruned_wts,prune_pct) = self.pn.get_prune_summary()
        tf.print("zeros after pruning:" + str(total_pruned_wts.numpy()))
        
        self.lh.log(step, total_trainable_wts, total_pruned_wts, prune_pct)
        
    def __prune(self):
        
        self.pn.enable_pruning()
        
        if self.pruning_type == "neurons":
            self.pn.prune_neurons(self.model,self.target_prune_pct)
        elif self.pruning_type == "weights":
            self.pn.prune_weights(self.model,self.target_prune_pct)
        elif self.pruning_type == "absolute_weights":
            self.pn.prune_abs_wts(self.model,self.target_prune_pct)
        
        self.pruning_done = True
        self.pn.disable_neuron_update()
   
class IntervalPruningCallback(keras.callbacks.Callback):
    def __init__(self, model, pruning_type,
                 pruning_values, epoch_range,
                 reset_neuron_count,
                 prune_dir, file_name):
        """ Save params in constructor
        """
        self.model = model
        self.pruning_type = pruning_type
        self.pruning_values = pruning_values
        self.epoch_range = epoch_range
        self.total_idx = len(self.epoch_range)
        self.idx = 0
        self.reset_neuron_count = reset_neuron_count
        self.pn = pn.PruneNetwork(model)
        model.set_prune_network(self.pn)
        self.lh = LogHandler(prune_dir, file_name)
    
    def on_train_begin(self, logs=None):
        self.pn.enable_neuron_update()
        #total_epochs = self.model.history.params["epochs"]
        #self.prune_at_interval = self.pruning_range/(self.pruning_intervals+1)
        #self.pruning_pct = self.target_pruning_pct/self.pruning_intervals
        
    def on_epoch_begin(self, epoch, logs=None):
        
        if self.idx == self.total_idx:
            self.pn.disable_neuron_update()
            return
        
        prune_at_epoch = self.epoch_range[self.idx]
        if (epoch+1) != prune_at_epoch:
            return
        self.pn.enable_pruning()
        
        pruning_pct = self.pruning_values[self.idx]
        self.idx += 1
        
        num_zeros = self.pn.get_zeros()
        tf.print("zeros before pruning:" + str(num_zeros))
        
        if self.pruning_type == "neurons":
            self.pn.sparsify_neurons(self.model,self.pruning_pct)
        elif self.pruning_type == "weights":
            self.pn.prune_weights(self.model, pruning_pct)
        elif self.pruning_type == "absolute_weights":
            self.pn.sparsify_absolute_weights(self.model,self.pruning_pct)
        
        (total_trainable_wts,
         total_pruned_wts,prune_pct) = self.pn.get_prune_summary()
        tf.print("zeros after pruning:" + str(total_pruned_wts.numpy()))
        
        self.lh.log(epoch, total_trainable_wts, total_pruned_wts, prune_pct)
        
        if self.reset_neuron_count == True:
            self.pn.reset_neuron_count()
        

class OptimalPruningCallback(keras.callbacks.Callback):
    def __init__(self, model, pruning_type, epoch_pruning_interval,
                 num_pruning, reset_neuron_count, log_dir,
                 prune_dir, file_name):
        """ Save params in constructor
        """
        self.model = model
        self.pruning_type = pruning_type
        self.epoch_pruning_interval = epoch_pruning_interval
        self.num_pruning = num_pruning
        self.idx = 0
        
        self.reset_neuron_count = reset_neuron_count
        self.pn = pn.PruneNetwork(model)
        model.set_prune_network(self.pn)
        self.lh = LogHandler(prune_dir, file_name)
        
        log_dir = log_dir + "/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
        file_writer.set_as_default()
        
    def on_train_begin(self, logs=None):
        self.pn.enable_neuron_update()
    
    def on_train_end(self,logs=None):
        self.lh.write_to_file()
    
    def on_epoch_begin(self,epoch,logs=None):
        (total_trainable_wts,
         total_pruned_wts,prune_pct) = self.pn.get_prune_summary()
        tf.print("zeros after pruning:" + str(total_pruned_wts.numpy()))
        wts_remaining = total_trainable_wts - total_pruned_wts.numpy()
        tf.summary.scalar('pruning',data=wts_remaining, step=epoch)
        
    def on_epoch_end(self, epoch, logs=None):
        
        if self.idx == self.num_pruning:
            self.pn.disable_neuron_update()
            return
        
        if (epoch+1)%self.epoch_pruning_interval != 0:
            return
        
        self.idx += 1
        self.pn.enable_pruning()
        
        num_zeros = self.pn.get_zeros()
        tf.print("zeros before pruning:" + str(num_zeros))
        
        self.pn.prune_optimal_weights(self.model, self.pruning_type)
        
        #self.lh.log(epoch, total_trainable_wts, total_pruned_wts, prune_pct)
        
        if self.reset_neuron_count == True:
            self.pn.reset_neuron_count()
    
class StopCallback(keras.callbacks.Callback):
    def __init__(self, training_accuracy):
        self.training_accuracy = training_accuracy
        self.num_epochs = 0
        
    def on_epoch_end(self,epoch,logs=None):
        self.accuracy = logs["accuracy"]
        self.num_epochs = epoch+1
        if self.accuracy >= self.training_accuracy:
            self.model.stop_training = 1
   
    def get_num_epochs(self):
        return self.num_epochs