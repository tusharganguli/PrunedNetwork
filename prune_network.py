#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 18:03:11 2021

@author: tushar
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np

class PruneNetwork:
    def __init__(self, model):
        self.model = model
        self.total_pruned_wts = 0
        self.functors = []
        self.__create_functors()
        self.neuron = self.__create_neurons()
        
        # variable for prune_optimal
        self.pruning_rate = 1
        
    def __create_neurons(self):
        trainable_vars = self.model.trainable_variables
        neuron = []
    
        for idx in range(len(trainable_vars)):
            if "kernel" not in trainable_vars[idx].name:
                continue
            rows,cols = trainable_vars[idx].shape 
            neuron.append([0 for x in range(cols)])
        return neuron
    
    def enable_neuron_update(self):
        self.model.enable_neuron_update()
    
    def disable_neuron_update(self):
        self.model.disable_neuron_update()
        
    def enable_pruning(self):
        self.model.enable_pruning()
            
    """
    Prune the network by setting weight values to 0 by sorting the  
    weights first on kernel access then neuron frequency and then the smallest 
    weights
    """
    def prune_weights(self, model, pruning_pct):
        all_weights = []
        layer_lst = []
        data_lst = []
        dim_lst = []
        layer_cnt = 0
        total_len = 0
            
        for layer in model.layers:
            if not isinstance(layer,keras.layers.Dense):
                continue
            weights = layer.get_weights()
            all_weights.append(weights)
            
            # get the neuron freq 
            neuron_freq = self.neuron[layer_cnt].numpy()
            
            # get kernel weights
            kernel = weights[0]
            rows,cols = kernel.shape 
            total_len += rows*cols
            dim_lst.append([rows,cols])
            
            layer_lst.append(layer)
            
            data_lst.extend([(layer_cnt,neuron_freq[j], 
                            kernel[i][j], i,j) 
                           for i in range(rows) for j in range(cols)])

            
            layer_cnt += 1
        #data_lst = [x for x in data_lst if x[3] == 0]
        zero_wts = [t for t in data_lst if t[2] == 0]
        non_zero_wts = [t for t in data_lst if t[2] != 0]
        sorted_lst = sorted(non_zero_wts,key=lambda x: (x[1], x[2]))
        del non_zero_wts
        
        #kernel_vals = [tple[2] for tple in sorted_lst]
        #total_vals = len(kernel_vals)
        pruning_idx = tf.cast(total_len * (pruning_pct/100),dtype=tf.int32)
                
        sorted_arr = np.array(sorted_lst)
        del sorted_lst
        sorted_arr[:pruning_idx,2] = 0
        
        new_lst = sorted_arr.tolist()
        new_lst.extend(zero_wts)
        del zero_wts
        del sorted_arr
        new_sorted_lst = sorted(new_lst,key=lambda x: (x[0],x[3], x[4]))
        #new_sorted_arr = np.array(new_sorted_lst)
        
        # retrieve all kernel values from the sorted list
        kernel_wts = [tple[2] for tple in new_sorted_lst]
        
        # convert the lists into arrays
        kernel_arr = np.array(kernel_wts)
        del kernel_wts
        
        start_offset = 0
        for idx in range(len(layer_lst)):
            dim = dim_lst[idx]
            total_elements = dim[0] * dim[1]
            end_offset = start_offset+total_elements
            kernel = np.reshape(kernel_arr[start_offset:end_offset],(dim))
            start_offset = end_offset
            all_weights[idx][0] = kernel
            del kernel
            layer_lst[idx].set_weights(all_weights[idx])      


    """
    Sparsify the network by setting weight values to 0 by sorting the  
    weights first on kernel access then neuron frequency and then the smallest 
    weights
    """
    def prune_absolute_weights(self, model, pruning_pct):
        self.__pruning_enabled()
        
        all_weights = []
        layer_lst = []
        data_lst = []
        dim_lst = []
        layer_cnt = 0
        total_len = 0
        
        for layer in model.layers:
            if not isinstance(layer,keras.layers.Dense):
                continue
            weights = layer.get_weights()
            all_weights.append(weights)
            
            # get the neuron freq 
            neuron_freq = self.neuron[layer_cnt].numpy()
            
            # get kernel weights
            kernel = weights[0]
            abs_kernel = np.absolute(kernel)
            rows,cols = kernel.shape 
            total_len += rows*cols
            dim_lst.append([rows,cols])
            
            layer_lst.append(layer)
            
            data_lst.extend([(layer_cnt,neuron_freq[j], 
                            kernel[i][j],abs_kernel[i][j], i,j) 
                           for i in range(rows) for j in range(cols)])

            
            layer_cnt += 1
        #data_lst = [x for x in data_lst if x[3] == 0]
        zero_wts = [t for t in data_lst if t[2] == 0]
        non_zero_wts = [t for t in data_lst if t[2] != 0]
        sorted_lst = sorted(non_zero_wts,key=lambda x: (x[1], x[3]))
        del non_zero_wts
        
        kernel_vals = [tple[2] for tple in sorted_lst]
        total_vals = len(kernel_vals)
        pruning_idx = tf.cast(total_vals * (pruning_pct/100),dtype=tf.int32)
                
        sorted_arr = np.array(sorted_lst)
        del sorted_lst
        sorted_arr[:pruning_idx,2] = 0
        
        new_lst = sorted_arr.tolist()
        new_lst.extend(zero_wts)
        del zero_wts
        del sorted_arr
        new_sorted_lst = sorted(new_lst,key=lambda x: (x[0],x[3], x[4]))
        
        # retrieve all kernel values from the sorted list
        kernel_wts = [tple[2] for tple in new_sorted_lst]
        
        # convert the lists into arrays
        kernel_arr = np.array(kernel_wts)
        del kernel_wts
        
        start_offset = 0
        for idx in range(len(layer_lst)):
            dim = dim_lst[idx]
            total_elements = dim[0] * dim[1]
            end_offset = start_offset+total_elements
            kernel = np.reshape(kernel_arr[start_offset:end_offset],(dim))
            start_offset = end_offset
            all_weights[idx][0] = kernel
            del kernel
            layer_lst[idx].set_weights(all_weights[idx])      

    """
    Prune the network by setting weight values to 0 by sorting the  
    weights first on kernel access then neuron frequency and then the smallest 
    weights
    """
    def prune_optimal_weights(self, model, pruning_type):
        
        network_pruned = False
        #while network_pruned != True:
        
        network_pruned = self.__prune_layers(model)
        
        #        self.pruning_rate -= 0.01 
            

    def __prune_layers(self, model):
        layer_cnt = 0
        network_pruned = False
        for layer in model.layers:
                if not isinstance(layer,keras.layers.Dense):
                    continue
                if layer.name == "output":
                    continue
                weights = layer.get_weights()
                
                # get the neuron freq 
                neuron_freq = self.neuron[layer_cnt].numpy()
                pruning_range = np.linspace(3,1,5)
                for pruning_rate in pruning_range:
                    kernel,pruning = self.__prune_avg(weights, neuron_freq,
                                                  pruning_rate)
                    if pruning == True:
                        tf.print("Layer:", layer.name,", Pruning Rate:", pruning_rate )
                        weights[0] = kernel
                        layer.set_weights(weights)      
                        layer_cnt += 1
                        network_pruned = True       
                        break
        return network_pruned                    
                        
    def __prune_avg(self, weights,neuron_freq,pruning_rate ):
        pruning = False
        # generate the minimum value lower than which all neurons 
        # are to be removed
        min_value = np.mean(neuron_freq)-(pruning_rate*np.std(neuron_freq))
        # get kernel weights
        kernel = weights[0]
        rows,cols = kernel.shape 
        for idx in range(cols):
            is_zero = np.all(kernel[:,idx] == 0)
            if is_zero == False and neuron_freq[idx] < min_value:
                pruning = True
                kernel[:,idx] = 0
        return (kernel,pruning)
    
    def __prune_max(self, weights,neuron_freq ):
        # generate the minimum value lower than which all neurons 
        # are to be removed
        min_value = np.max(neuron_freq)-(0.9*np.std(neuron_freq))
        # get kernel weights
        kernel = weights[0]
        rows,cols = kernel.shape 
        for idx in range(cols):
            is_zero = np.all(kernel[:,idx] == 0)
            if is_zero == False and neuron_freq[idx] < min_value:
                kernel[:,idx] = 0
        return kernel
    
    def __create_functors(self):
        inp = self.model.input
        outputs = []
        
        for layer in self.model.layers:
            if not isinstance(layer,keras.layers.Dense):
                continue
            outputs += [layer.output]
        # evaluation functions
        self.functors = [K.function([inp], [out]) for out in outputs]    
    
    def update_neuron_frequency(self,x):
        count = 0
        for func in self.functors:
            activation_data = func(x)
            self.__update_frequency(count,activation_data[0])
            count += 1
    
    def __update_frequency( self, idx, activation_data):
        
        activation_dim = activation_data.shape
        for step in range(0,activation_dim[0]):
            condition = tf.equal(activation_data[step], 0)
            res_1 = tf.add(self.neuron[idx],1)
            self.neuron[idx] = tf.where(condition, self.neuron[idx],res_1)
     
    def get_zeros(self):
        wts = self.model.trainable_variables
        num_zeros = 0
        for w in wts:
            if "kernel" in w.name:
                kernel = w.numpy()
                num_zeros += kernel[np.where(kernel == 0)].size
        return num_zeros
    
    def get_prune_summary(self):
        trainable_wts_cnt = 0
        trainable_wts = self.model.trainable_weights
        pruned_wts_cnt = 0
        for wts in trainable_wts:
            if "kernel" in wts.name:
                trainable_wts_cnt = tf.add(trainable_wts_cnt,tf.size(wts))
                kernel_arr = wts.numpy()
                num_zeros = tf.size(kernel_arr[np.where(kernel_arr == 0)])
                pruned_wts_cnt = tf.add(pruned_wts_cnt,num_zeros)
        
        tf.print("Trainable variables:",trainable_wts_cnt)
        tf.print("Variables pruned:",pruned_wts_cnt)
        prune_pct = (pruned_wts_cnt/trainable_wts_cnt)*100
        tf.print("Prune percentage:",prune_pct)
        return (trainable_wts_cnt,pruned_wts_cnt,prune_pct)
    
    def reset_neuron_count(self):
        for idx in range(len(self.neuron)):
            zeros = tf.zeros(self.neuron[idx].shape,dtype=tf.int32)
            self.neuron[idx] = zeros
    
        
"""
    def sparsify_neurons(self, model, pruning_pct):
            myList = []
            sorted_lst = []
            total_len = 0
            all_weights = []
            layer_cnt = 0
            neuron_len = []
            for layer in model.layers:
                if isinstance(layer,keras.layers.Dense) == True:
                    weights = layer.get_weights()
                    all_weights.append(weights)
                    neuron_len.append(weights[2].shape[0])
                    total_len += neuron_len[-1]
                    myList.extend([(layer_cnt,layer,i,weights[2][i]) for i in range(neuron_len[-1])])
                    layer_cnt += 1
            sorted_lst = sorted(myList,key=lambda x: (x[3]))
            bottom_n = tf.cast(total_len * (pruning_pct/100),dtype=tf.int32)
            sorted_arr = np.array(sorted_lst)
            sorted_arr[0:bottom_n,3] = 0
            sorted_lst = sorted(sorted_arr,key=lambda x: (x[0],x[2]))
            sorted_arr = np.array(sorted_lst)
            start = 0
            for idx in range(layer_cnt):
                end = start+neuron_len[idx]
                # initialize the weights to be removed to 0
                all_weights[idx][0][:,sorted_arr[start:end,3] == 0] = 0
                all_weights[idx][2][:] = 0 #sorted_arr[start:end,3]
                sorted_lst[start][1].set_weights(all_weights[idx])
                start = end
    
    def sparsify_neuron_weights(self, model, pruning_pct):
        total_len = 0
        all_weights = []
        layer_cnt = 0
        layer_list = []
        neuron_wts_lst = []
        wts_lst = []
        neuron_dim = []
        
        for layer in model.layers:
            if isinstance(layer,cl.MyDense) == True:
                weights = layer.get_weights()
                dim = weights[2].shape[0]
                neuron_wts = (weights[0].T * np.reshape(weights[2],(dim,1))).T
                neuron_wts = neuron_wts.flatten()
                neuron_wts_lst.extend(neuron_wts)
                wts_lst.extend(weights[0].flatten())
                neuron_dim.append(weights[0].shape)
                total_len += weights[0].shape[0]*weights[0].shape[1]
                layer_list.append(layer)
                all_weights.append(weights)
                layer_cnt += 1
        bottom_n = tf.cast(total_len * (pruning_pct/100),dtype=tf.int32)
        neuron_wts_arr = np.array(neuron_wts_lst)
        idx = neuron_wts_arr.argsort()[:bottom_n]
        wts_arr = np.array(wts_lst)
        wts_arr[idx] = 0
        start = 0
        offset = 0
        for idx in range(len(layer_list)):
            dim = neuron_dim[idx]
            offset += neuron_dim[idx][0]* neuron_dim[idx][1]
            wts = np.reshape(wts_arr[start:offset],dim)
            all_weights[idx][0] = wts
            all_weights[idx][2][:] = 0
            start = offset
            layer_list[idx].set_weights(all_weights[idx])
"""