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
import random

import prune_network as pn

class PruningCallback(keras.callbacks.Callback):
    def __init__(self, model, train_data, pruning_type,
                 pruning_pct, pruning_chg, 
                 prune_accuracy_threshold, prune_freq,
                 reset_neuron_count,
                 neuron_ctr_start_at_acc,
                 prune_dir, file_name):
        """ Save params in constructor
        """
        self.train_data = train_data
        self.pruning_type = pruning_type
        self.pruning_pct = pruning_pct
        self.pruning_chg = pruning_chg
        self.prune_accuracy_threshold = prune_accuracy_threshold
        self.prune_freq = prune_freq
        self.reset_neuron_count =  reset_neuron_count
        self.accuracy = 0
        self.count = 0
        self.pn = pn.PruneNetwork(model)
        model.set_prune_network(self.pn)
        self.prev_loss = 1
        self.df = pd.DataFrame(columns = ['Epoch', 
                                          'Total Trainable Wts',
                                          'Total Pruned Wts',
                                          'Prune Percentage'
                                          ])
        self.prune_dir = prune_dir
        if not os.path.exists(self.prune_dir):
            os.makedirs(self.prune_dir)
        self.file_name = file_name
        
        self.neuron_ctr_start_at_acc = neuron_ctr_start_at_acc
        
    
    def on_epoch_begin(self, epoch, logs=None):
        if self.accuracy < self.prune_accuracy_threshold:
            return
        self.count += 1
        
        if self.count%self.prune_freq != 0:
            return
        
        num_zeros = self.pn.get_zeros()
        tf.print("zeros before sparsification:" + str(num_zeros))
        
        if self.pruning_type == "neurons":
            self.pn.sparsify_neurons(self.model,self.pruning_pct)
        elif self.pruning_type == "weights":
            self.pn.sparsify_weights(self.model,self.pruning_pct)
        elif self.pruning_type == "absolute_weights":
            self.pn.sparsify_absolute_weights(self.model,self.pruning_pct)
        
        (total_trainable_wts,
         total_sparsed_wts,sparse_pct) = self.pn.get_prune_summary()
        tf.print("zeros after sparsification:" + str(total_sparsed_wts.numpy()))
        data = [epoch,total_trainable_wts.numpy(),
                total_sparsed_wts.numpy(),
                sparse_pct.numpy()]
        df2 = pd.DataFrame([data], columns=list(self.df))
        self.df = self.df.append(df2,ignore_index = True)
        
        self.pruning_pct += self.pruning_chg
        
        if self.reset_neuron_count == True:
            self.pn.reset_neuron_count()
    
    def on_epoch_end(self,epoch,logs=None):
        pass
        
            
    def on_train_end(self,logs=None):
        self.__write_to_file()
        
    def on_batch_end(self,batch,logs=None):
        self.accuracy = logs["accuracy"]
        # update the neuron frequency
        if self.accuracy >= self.neuron_ctr_start_at_acc:
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
    def __write_to_file(self):
        writer = pd.ExcelWriter(self.prune_dir + self.file_name + ".xls")
        self.df.to_excel(writer)
        # save the excel
        writer.save()
        
    
    
class StopCallback(keras.callbacks.Callback):
    def __init__(self, training_accuracy):
        self.training_accuracy = training_accuracy
        
    def on_epoch_end(self,epoch,logs=None):
        self.accuracy = logs["accuracy"]
        if self.accuracy >= self.training_accuracy:
            self.model.stop_training = 1
    