#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 21:37:06 2021

@author: tushar
"""

import tensorflow as tf
from tensorflow import keras
import pandas as pd
from datetime import datetime

import prune_network as pn
import generate_plots as gp
import utils

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
        
class CIPCallback(keras.callbacks.Callback):
    def __init__(self, model,
                 neuron_update_at_acc,
                 prune_start_at_acc,
                 num_pruning,
                 final_training_acc,
                 target_prune_pct,
                 prune_dir, file_name):
        """ Save params in constructor
        """
        self.model = model
        self.pruning_done = False
        self.neuron_update_at_acc = neuron_update_at_acc
        self.prune_start_at_acc = prune_start_at_acc
        self.num_pruning = num_pruning
        self.final_training_acc = final_training_acc
        self.prune_acc_interval = 0
        self.target_prune_pct = target_prune_pct
        self.interval_prune_pct = 0
        self.pn = pn.PruneNetwork(model)
        model.set_prune_network(self.pn)
        self.lh = utils.LogHandler(prune_dir, file_name)
        self.prune_dir = prune_dir
        #self.first_time = 1
        
    def on_train_begin(self, logs=None):
        self.prune_acc_interval = \
            (self.final_training_acc - self.prune_start_at_acc)/self.num_pruning
        self.interval_prune_pct = self.target_prune_pct / self.num_pruning
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
        if prune_pct.numpy() >= self.target_prune_pct:
            self.pruning_done = True
            self.pn.disable_neuron_update()
            #self.pn.write_svd()

        self.prune_start_at_acc += self.prune_acc_interval
        
    def on_train_end(self,logs=None):
        svd_plots = gp.SVDPlots()
        [svd_df,svd_plot_info,layer_cnt] = self.pn.GetSVDDetails()
        
        final_svd = self.get_final_sv()
        final_acc = logs["accuracy"]
        svd_plots.PlotRatio(svd_df, svd_plot_info, layer_cnt, 
                            final_svd, final_acc, self.prune_dir)
        
        self.pn.ConvertSVDPlots(self.prune_dir)
    
    def get_final_sv(self):
        from scipy.linalg import svd
        
        svd_df = pd.DataFrame()
        
        for layer in self.model.layers:
            if not isinstance(layer,keras.layers.Dense):
                continue
            weights = layer.get_weights()
            u,s,vt = svd(weights[0])
            df_s = pd.DataFrame(s)
            svd_df = pd.concat([svd_df,df_s], ignore_index=True, axis=1)
            
        return svd_df
    
    def __prune_and_log(self, step, accuracy):
        num_zeros = self.pn.get_zeros()
        tf.print("zeros before pruning:" + str(num_zeros))
        
        self.pn.enable_pruning()
            
        self.pn.cip(self.model,self.interval_prune_pct, accuracy)
        
        (total_trainable_wts,
         total_pruned_wts,prune_pct) = self.pn.get_prune_summary()
        
        
        tf.print("zeros after pruning:" + str(total_pruned_wts.numpy()))
        
        self.lh.log(step, total_trainable_wts, total_pruned_wts, prune_pct)
        
        
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

class BasePruneCallback(keras.callbacks.Callback):
    def __init__(self, model, total_prune_pct, final_acc, reset_neuron,
               log_dir, prune_dir, log_file_name):
        
        self.model = model
        self.pruning_done = False
        self.pn = pn.PruneNetwork(model)
        model.set_prune_network(self.pn)
        self.total_prune_pct = total_prune_pct
        self.final_acc = final_acc
        self.reset_neuron = reset_neuron
        self.log_dir = log_dir
        self.prune_dir = prune_dir
        self.log_file_name = log_file_name
        self.lh = utils.LogHandler(prune_dir, log_file_name)
        self.num_epochs = 0
        
    def __del__(self):
        del self.model
        del self.total_prune_pct
        del self.final_acc
        del self.reset_neuron
        del self.log_dir
        del self.prune_dir
        del self.log_file_name
        del self.lh
    
    def on_train_end(self,logs=None):
        self.lh.write_to_file()
    
    def on_epoch_end(self, epoch, logs=None):
        self.num_epochs = epoch+1
        
    def get_num_epochs(self):
        return self.num_epochs
    
class PruneTrainedCallback(BasePruneCallback):
    def __init__(self, model, num_pruning, pruning_interval,
               total_prune_pct, final_acc, reset_neuron,
               log_dir, prune_dir, log_file_name):
        super(PruneTrainedCallback, self).__init__(model, total_prune_pct, 
                                                    final_acc, reset_neuron,
                                                    log_dir, prune_dir, log_file_name)
        
        self.num_pruning = num_pruning
        self.pruning_interval = pruning_interval
        self.interval_cnt = self.pruning_interval
        self.pruning_cnt = 0
        self.total_prune_pct = total_prune_pct
        # amount of pruning to be carried out in 1 cycle of pruning
        self.interval_prune_pct = self.total_prune_pct/self.num_pruning
        
        
    def __del__(self):
        del self.num_pruning
        del self.pruning_interval
        
    def on_train_begin(self, logs=None):
        self.pn.enable_neuron_update()
    
    def __prune_and_log(self, step, accuracy):
        num_zeros = self.pn.get_zeros()
        tf.print("zeros before pruning:" + str(num_zeros))
        
        self.pn.enable_pruning()
            
        self.pn.cip(self.model,self.interval_prune_pct, accuracy)
        
        (total_trainable_wts,
         total_pruned_wts,prune_pct) = self.pn.get_prune_summary()
        
        
        tf.print("zeros after pruning:" + str(total_pruned_wts.numpy()))
        
        self.lh.log(step, total_trainable_wts, total_pruned_wts, prune_pct)
        
    def on_epoch_end(self, epoch, logs=None):
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
        
        if self.pruning_cnt == self.num_pruning:
            self.pruning_done = True
            self.pn.disable_neuron_update()
            
        
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