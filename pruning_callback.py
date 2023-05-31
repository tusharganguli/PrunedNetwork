#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 21:37:06 2021

@author: tushar
"""

import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import numpy as np
import random

import prune_network as pn
import generate_plots as gp
import utils

class BasePruneCallback(keras.callbacks.Callback):
    def __init__(self, model, prune_pct, final_acc, reset_neuron,
                 log_handler):
        
        self.model = model
        self.pruning_done = False
        self.pn = pn.PruneNetwork(model)
        model.set_prune_network(self.pn)
        self.prune_pct = prune_pct
        self.final_acc = final_acc
        self.reset_neuron = reset_neuron
        self.lh = log_handler
        self.num_epochs = 0
        
    def __del__(self):
        del self.model
        del self.prune_pct
        del self.final_acc
        del self.reset_neuron
        del self.lh
    
    def on_train_end(self,logs=None):
        #self.lh.write_to_file()
        pass
    
    def on_epoch_end(self, epoch, logs=None):
        self.num_epochs = epoch+1
        
    def get_num_epochs(self):
        return self.num_epochs
    
    def update_neuron(self):
        rng = np.random.default_rng()
        data = rng.choice(self.train_img,32)
        self.pn.update_neuron_frequency(data)
    

class CNNCallback(BasePruneCallback):
    def __init__(self, model, train_img, prune_start_at_acc,
                 num_pruning, final_acc, prune_pct, neuron_update,
                 pruning_type, reset_neuron, log_handler, tb_log_filename ):
        super(CNNCallback, self).__init__(model, prune_pct, 
                                          final_acc, reset_neuron,
                                          log_handler)
        """ Save params in constructor
        """
        self.train_img = train_img
        self.prune_start_at_acc = prune_start_at_acc
        self.num_pruning = num_pruning
        self.prune_acc_interval = 0
        self.prune_pct_lst = []
        self.prune_pct_idx = 0
        self.neuron_update = neuron_update
        self.pruning_type = pruning_type
        self.plot_dir = log_handler.get_plot_dir()
        self.tb_log_filename = tb_log_filename
        
        self.prune_summary_writer = tf.summary.create_file_writer(self.tb_log_filename+ "/prune_pct")
        #logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        #file_writer = tf.summary.create_file_writer(logdir + "/metrics")
        self.prune_summary_writer.set_as_default()
        
    
    def __del__(self):
        del self.prune_start_at_acc
        del self.num_pruning
        del self.prune_acc_interval
        del self.prune_pct_lst
        del self.prune_pct_idx
        del self.neuron_update
        del self.pruning_type
        del self.prune_summary_writer
        
    def on_train_begin(self, logs=None):
        self.pn.enable_neuron_update(self.neuron_update)
        
        self.prune_acc_interval = \
            (self.final_acc - self.prune_start_at_acc)/self.num_pruning
        self.prune_pct_lst = utils.get_exp_decay_range(self.num_pruning)
        self.prune_pct_lst = self.prune_pct_lst*(self.prune_pct/100)
    
    def on_train_end(self,logs=None):
        svd_plots = gp.SVDPlots()
        [sig_df,svd_plot_info, layer_cnt] = self.lh.get_svd_details(self.model)
        
        final_sv = self.lh.get_sv(self.model)
        final_acc = logs["accuracy"]
        #svd_plots.PlotRatio(sig_df, svd_plot_info, layer_cnt, 
        #                    final_sv, final_acc, self.plot_dir)
        
        #svd_plots.ConvertToEps(self.plot_dir)
        self.lh.reset_svd()
            
    def __log_prune_tb(self, epoch):
        (total_trainable_wts,
         total_pruned_wts,prune_pct) = self.pn.get_prune_summary()
        remaining_pct_wts = 1-prune_pct.numpy()
        tf.summary.scalar('prune_pct', data=remaining_pct_wts, step=epoch)        
        
    def __prune_and_log(self, epoch, accuracy):
        num_zeros = self.pn.get_zeros()
        #tf.print("zeros before pruning:" + str(num_zeros))
        
        self.pn.enable_pruning()
        
        self.lh.log_sv(self.model)
        
        prune_pct = self.prune_pct_lst[self.prune_pct_idx]
        self.pn.prune_cnn(prune_pct, self.pruning_type)
        self.prune_pct_idx += 1
        
        (total_trainable_wts,
         total_pruned_wts,prune_pct) = self.pn.get_prune_summary()
        
        #remaining_pct_wts = 1-prune_pct.numpy()
        #tf.summary.scalar('prune_pct', data=remaining_pct_wts, step=epoch)        
        
        self.lh.log_sv(self.model)
        self.lh.log_prune_info(prune_pct, accuracy)
        
        #tf.print("zeros after pruning:" + str(total_pruned_wts.numpy()))
    
    
    def on_epoch_end(self, epoch, logs=None):
        super(CNNCallback, self).on_epoch_end(epoch, logs)
        
        self.__log_prune_tb(epoch)
        if self.pruning_done == True:
            return
        
        if self.pn.neuron_update_flag() == True:
            self.update_neuron()
            
        accuracy = logs["accuracy"]
        if accuracy < self.prune_start_at_acc:
            return
        
        self.__prune_and_log(epoch,accuracy)
        
        #if prune_pct.numpy() >= self.prune_pct:
        if self.num_pruning == self.prune_pct_idx:
            self.pruning_done = True
            self.pn.disable_neuron_update()
            #self.lh.write_svd()

        self.prune_start_at_acc += self.prune_acc_interval
        
        


class CIPCallback(BasePruneCallback):
    def __init__(self, model, train_img, prune_start_at_acc,
                 num_pruning, final_acc, prune_pct, neuron_update,
                 pruning_type, reset_neuron, log_handler ):
        super(CIPCallback, self).__init__(model, prune_pct, 
                                          final_acc, reset_neuron,
                                          log_handler)
        """ Save params in constructor
        """
        self.train_img = train_img
        self.prune_start_at_acc = prune_start_at_acc
        self.num_pruning = num_pruning
        self.prune_acc_interval = 0
        self.prune_pct_lst = []
        self.prune_pct_idx = 0
        self.neuron_update = neuron_update
        self.pruning_type = pruning_type
        self.plot_dir = log_handler.get_plot_dir()
    
    def __del__(self):
        del self.prune_start_at_acc
        del self.num_pruning
        del self.prune_acc_interval
        del self.prune_pct_lst
        del self.prune_pct_idx
        del self.neuron_update
        del self.pruning_type
    
    def on_train_begin(self, logs=None):
        self.pn.enable_neuron_update(self.neuron_update)
        
        if self.num_pruning == 0:
            self.prune_acc_interval = 0
        else:
            self.prune_acc_interval = \
                (self.final_acc - self.prune_start_at_acc)/self.num_pruning
        self.prune_pct_lst = utils.get_exp_decay_range(self.num_pruning)
        self.prune_pct_lst = self.prune_pct_lst*(self.prune_pct/100)
    
    def on_train_end(self,logs=None):
        svd_plots = gp.SVDPlots()
        [sig_df,svd_plot_info, layer_cnt] = self.lh.get_svd_details(self.model)
        
        final_sv = self.lh.get_sv(self.model)
        final_acc = logs["accuracy"]
        svd_plots.PlotRatio(sig_df, svd_plot_info, layer_cnt, 
                            final_sv, final_acc, self.plot_dir)
        
        svd_plots.ConvertToEps(self.plot_dir)
        self.lh.reset_svd()
    
    def __prune_and_log(self, step, accuracy):
        num_zeros = self.pn.get_zeros()
        tf.print("zeros before pruning:" + str(num_zeros))
        
        self.pn.enable_pruning()
        
        self.lh.log_sv(self.model)
        
        prune_pct = self.prune_pct_lst[self.prune_pct_idx]
        self.pn.cip(prune_pct, self.pruning_type)
        self.prune_pct_idx += 1
        
        (total_trainable_wts,
         total_pruned_wts,prune_pct) = self.pn.get_prune_summary()
        
        self.lh.log_sv(self.model)
        self.lh.log_prune_info(prune_pct, accuracy)
        
        tf.print("zeros after pruning:" + str(total_pruned_wts.numpy()))
            
    def on_epoch_end(self, epoch, logs=None):
        super(CIPCallback, self).on_epoch_end(epoch, logs)
        accuracy = logs["accuracy"]
        
        if self.pn.neuron_update_flag() == True:
            self.update_neuron()
        
        if self.pruning_done == True:
            if accuracy >= self.final_acc:
                self.model.stop_training = True
            return
        
        if accuracy < self.prune_start_at_acc:
            return
        
        self.__prune_and_log(epoch,accuracy)
        
        if self.reset_neuron == True:
            self.pn.reset_neuron_count()
        
        #if prune_pct.numpy() >= self.prune_pct:
        if self.num_pruning == self.prune_pct_idx:
            self.pruning_done = True
            self.pn.disable_neuron_update()
            #self.lh.write_svd()

        self.prune_start_at_acc += self.prune_acc_interval

class RATCallback(BasePruneCallback):
    def __init__(self, model, final_acc, log_handler):    
        prune_pct = 0
        reset_neuron = False
        
        super(PruneTrainedCallback, self).__init__(model, prune_pct, 
                                                   final_acc, reset_neuron,
                                                   log_handler)
    def on_train_begin(self, logs=None):
        """
        We want to preserve the weights which have been initialized to 0. 

        """
        self.model.disable_neuron_update()
        self.model.preserve_pruning()
       
       
       
class PruneTrainedCallback(BasePruneCallback):
    def __init__(self, model, num_pruning, final_acc, 
                 prune_pct, neuron_update, prune_type,
                 reset_neuron, log_handler):
        super(PruneTrainedCallback, self).__init__(model, prune_pct, 
                                                   final_acc, reset_neuron,
                                                   log_handler)
        self.num_pruning = num_pruning
        self.prune_acc_interval = 0
        self.prune_pct_lst = []
        self.prune_pct_idx = 0
        self.neuron_update = neuron_update
        self.prune_type = prune_type
        self.pruning_cnt = 0
        self.plot_dir = log_handler.get_plot_dir()
        
    def __del__(self):
        del self.num_pruning
        del self.prune_acc_interval
        del self.prune_pct_lst
        del self.prune_pct_idx
        del self.neuron_update
        del self.prune_type
        del self.pruning_cnt
        
        
    def on_train_begin(self, logs=None):
        self.pn.enable_neuron_update(self.neuron_update)
        self.prune_acc_interval = \
            (self.final_acc - self.prune_start_at_acc)/self.num_pruning
        self.prune_pct_lst = utils.get_exp_decay_range(self.num_pruning)
        self.prune_pct_lst = self.prune_pct_lst*(self.prune_pct/100)
    
    def on_train_end(self,logs=None):
        svd_plots = gp.SVDPlots()
        [sig_df,svd_plot_info, layer_cnt] = self.lh.get_svd_details(self.model)
        
        final_sv = self.lh.get_sv(self.model)
        final_acc = logs["accuracy"]
        svd_plots.PlotRatio(sig_df, svd_plot_info, layer_cnt, 
                            final_sv, final_acc, self.plot_dir)
        
        svd_plots.ConvertToEps(self.plot_dir)
        self.lh.reset_svd()
        
    def __prune_and_log(self, step, accuracy):
        num_zeros = self.pn.get_zeros()
        tf.print("zeros before pruning:" + str(num_zeros))
        
        self.pn.enable_pruning()
        
        self.lh.log_sv(self.model)
        
        self.pn.cip(self.interval_prune_pct, self.prune_type)
        
        (total_trainable_wts,
         total_pruned_wts,prune_pct) = self.pn.get_prune_summary()
        
        self.lh.log_sv(self.model)
        self.lh.log_prune_info(prune_pct, accuracy)
        
        tf.print("zeros after pruning:" + str(total_pruned_wts.numpy()))
        
        
    def on_epoch_end(self, epoch, logs=None):
        super(PruneTrainedCallback, self).on_epoch_end(epoch, logs)
        accuracy = logs["accuracy"]
        
        if self.pruning_cnt == self.num_pruning and accuracy >= self.final_acc:
            self.model.stop_training = 1
            return 
        
        if self.pruning_done == True:
            return
        
        if self.interval_cnt != self.pruning_interval:
            self.interval_cnt += 1
            return
        
        self.interval_cnt = 0
        self.pruning_cnt += 1
        
        self.__prune_and_log(epoch,accuracy)
        
        if self.reset_neuron == True:
            self.pn.reset_neuron_count()
        
        
        if self.pruning_cnt == self.num_pruning:
            self.pruning_done = True
            self.pn.disable_neuron_update()
            
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
        self.lh = utils.LogHandler(prune_dir, file_name)
        
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
    
class EarlyStoppingCallback(keras.callbacks.Callback):
    def __init__(self, delta=0, verbose=0):
        self.delta = delta
        self.verbose = verbose
        self.min_val_loss = np.Inf
        self.tick = 0
        self.prev_loss = 0
        
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs["val_loss"]
        
        if self.verbose == 1:
            tf.print("Min Val Loss:", self.min_val_loss)
            tf.print("Current Val Loss:", val_loss)
            
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
        elif val_loss > (1+self.delta)*self.min_val_loss and self.prev_loss < val_loss:
            self.tick += 1
        elif val_loss < (1+self.delta)*self.min_val_loss:
            self.tick = 0
        if self.tick == 3:
            self.model.stop_training = True
        self.prev_loss = val_loss
        
