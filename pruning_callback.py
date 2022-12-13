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

class CNNCallback(BasePruneCallback):
    def __init__(self, model, prune_start_at_acc,
                 num_pruning, final_acc, prune_pct, neuron_update,
                 pruning_type, reset_neuron, log_handler ):
        super(CNNCallback, self).__init__(model, prune_pct, 
                                          final_acc, reset_neuron,
                                          log_handler)
        """ Save params in constructor
        """
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
        self.pn.cnn(prune_pct, self.pruning_type)
        self.prune_pct_idx += 1
        
        (total_trainable_wts,
         total_pruned_wts,prune_pct) = self.pn.get_prune_summary()
        
        self.lh.log_sv(self.model)
        self.lh.log_prune_info(prune_pct, accuracy)
        
        tf.print("zeros after pruning:" + str(total_pruned_wts.numpy()))
            
    def on_epoch_end(self, epoch, logs=None):
        super(CNNCallback, self).on_epoch_end(epoch, logs)
        
        if self.pruning_done == True:
            return
        
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
    def __init__(self, model, prune_start_at_acc,
                 num_pruning, final_acc, prune_pct, neuron_update,
                 pruning_type, reset_neuron, log_handler ):
        super(CIPCallback, self).__init__(model, prune_pct, 
                                          final_acc, reset_neuron,
                                          log_handler)
        """ Save params in constructor
        """
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
        
"""
class FWPCallback(BasePruneCallback):
    def __init__(self, model,
                 neuron_update_at_acc,
                 prune_start_at_acc,
                 num_pruning,
                 reset_neuron,
                 final_training_acc,
                 total_prune_pct,
                 prune_dir, log_file_name): 
        super(FWPCallback, self).__init__(model, total_prune_pct, 
                                          final_training_acc, reset_neuron,
                                          prune_dir, log_file_name)
        
        # Save params in constructor
        
        self.model = model
        self.pruning_done = False
        self.neuron_update_at_acc = neuron_update_at_acc
        self.prune_start_at_acc = prune_start_at_acc
        self.num_pruning = num_pruning
        self.final_training_acc = final_training_acc
        self.prune_acc_interval = 0
        self.total_prune_pct = total_prune_pct
        self.interval_prune_pct = 0
        self.pn = pn.PruneNetwork(model)
        model.set_prune_network(self.pn)
        self.lh = utils.LogHandler(prune_dir, log_file_name)
        self.prune_dir = prune_dir
        
    def on_train_begin(self, logs=None):
        self.prune_acc_interval = \
            (self.final_training_acc - self.prune_start_at_acc)/self.num_pruning
        self.interval_prune_pct = self.total_prune_pct / self.num_pruning
        if self.neuron_update_at_acc == 1:
            self.pn.enable_neuron_update()
            
    def on_epoch_end(self, epoch, logs=None):
        if self.pruning_done == True:
            return
        
        accuracy = logs["accuracy"]
        
        if self.neuron_update_at_acc != 1:
            if accuracy < self.neuron_update_at_acc:
                return
            else:
                self.pn.enable_neuron_update()
                
        if accuracy < self.prune_start_at_acc:
            return
        
        self.__prune_and_log(epoch,accuracy)
        
        (trainable_wts_cnt,pruned_wts_cnt,prune_pct) = self.pn.get_prune_summary()
        if prune_pct.numpy() >= self.total_prune_pct:
            self.pruning_done = True
            self.pn.disable_neuron_update()
            #self.pn.write_svd()

        self.prune_start_at_acc += self.prune_acc_interval
        
    
    def __prune_and_log(self, step, accuracy):
        num_zeros = self.pn.get_zeros()
        tf.print("zeros before pruning:" + str(num_zeros))
        
        self.pn.enable_pruning()
            
        self.pn.fwp(self.model,self.interval_prune_pct, accuracy)
        
        (total_trainable_wts,
         total_pruned_wts,prune_pct) = self.pn.get_prune_summary()
        
        
        tf.print("zeros after pruning:" + str(total_pruned_wts.numpy()))
        
        self.lh.log(step, total_trainable_wts, total_pruned_wts, prune_pct)   
        

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
        self.lh = utils.LogHandler(prune_dir, log_file_name)
        self.prune = "on_batch_end"
        self.one_time_update = 0
        
    def on_epoch_end(self, epoch, logs=None):
        if self.pruning_done == True:
            return
        
        accuracy = logs["accuracy"]
        
        if self.neuron_update_at_acc == 1:
            if accuracy >= self.prune_at_accuracy and self.one_time_update == 0:
                self.pn.enable_neuron_update()
                self.one_time_update = 1
                return
        elif accuracy > self.neuron_update_at_acc and self.one_time_update == 0:
            self.pn.enable_neuron_update()
            self.one_time_update = 1
            return
         
        if accuracy < self.prune_at_accuracy:
            return 
        
        self.__prune_and_log(epoch,accuracy)
        
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


class PruningCallback(keras.callbacks.Callback):
    def __init__(self, model, train_data, pruning_type,
                 pruning_pct, pruning_chg, 
                 prune_at_accuracy, prune_freq,
                 reset_neuron_count,
                 neuron_start_at_acc,
                 prune_dir, file_name):
        #Save params in constructor
        
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
        self.lh = utils.LogHandler(prune_dir, file_name)
        
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


class IntervalPruningCallback(keras.callbacks.Callback):
    def __init__(self, model, pruning_type,
                 pruning_values, epoch_range,
                 reset_neuron_count,
                 prune_dir, file_name):
        #Save params in constructor
        
        self.model = model
        self.pruning_type = pruning_type
        self.pruning_values = pruning_values
        self.epoch_range = epoch_range
        self.total_idx = len(self.epoch_range)
        self.idx = 0
        self.reset_neuron_count = reset_neuron_count
        self.pn = pn.PruneNetwork(model)
        model.set_prune_network(self.pn)
        self.lh = utils.LogHandler(prune_dir, file_name)
    
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

"""