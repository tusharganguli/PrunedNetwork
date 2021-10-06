#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 18:03:11 2021

@author: tushar
"""
import tensorflow as tf
import numpy as np

import custom_layer as cl

class SparseNetwork:
    magic_number = 0
            
    def sparsify_neurons(self, model, pruning_pct):
        myList = []
        sorted_lst = []
        total_len = 0
        all_weights = []
        layer_cnt = 0
        neuron_len = []
        for layer in model.layers:
            if isinstance(layer,cl.MyDense) == True:
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
            
    '''
    Sparsify the network by setting weight values to 0 by sorting the  
    weights first on kernel access then neuron frequency and then the smallest 
    weights
    '''
    def sparsify_weights(self, model, pruning_pct):
        
        all_weights = []
        layer_lst = []
        data_lst = []
        dim_lst = []
        layer_cnt = 0
        total_len = 0
        for layer in model.layers:
            if isinstance(layer,cl.MyDense) == True:
                weights = layer.get_weights()
                all_weights.append(weights)
                
                # get the neuron freq 
                neuron_freq = weights[2]
                
                # get kernel weights
                kernel = weights[0]
                rows,cols = kernel.shape 
                total_len += rows*cols
                dim_lst.append([rows,cols])
                # get kernel access
                ka = weights[3]
                
                layer_lst.append(layer)
                
                data_lst.extend([(layer_cnt,neuron_freq[j], 
                                kernel[i][j], ka[i][j],i,j) 
                               for i in range(rows) for j in range(cols)])
                layer_cnt += 1
        #data_lst = [x for x in data_lst if x[3] == 0]
        sorted_lst = sorted(data_lst,key=lambda x: (x[3],x[1], x[2]))
        
        ka_col = [tple[3] for tple in sorted_lst]
        total_cols = ka_col.count(0)
        pruning_idx = tf.cast(total_cols * (pruning_pct/100),dtype=tf.int32)
        
        sorted_arr = np.array(sorted_lst)
        del sorted_lst
        sorted_arr[:pruning_idx,2] = 0
        sorted_arr[:pruning_idx,3] = 1
        
        new_lst = sorted_arr.tolist()
        del sorted_arr
        new_sorted_lst = sorted(new_lst,key=lambda x: (x[0],x[4], x[5]))
        #new_sorted_arr = np.array(new_sorted_lst)
        
        # retrieve all kernel and kernel access values from the sorted list
        kernel_wts = [tple[2] for tple in new_sorted_lst]
        kernel_access = [tple[3] for tple in new_sorted_lst]
        
        # convert the lists into arrays
        kernel_arr = np.array(kernel_wts)
        ka_arr = np.array(kernel_access, dtype='int32')
        del kernel_wts
        del kernel_access
        start_offset = 0
        for idx in range(len(layer_lst)):
            dim = dim_lst[idx]
            total_elements = dim[0] * dim[1]
            end_offset = start_offset+total_elements
            kernel = np.reshape(kernel_arr[start_offset:end_offset],(dim))
            kernel_access = np.reshape(ka_arr[start_offset:end_offset],(dim))
            start_offset = end_offset
            all_weights[idx][0] = kernel
            all_weights[idx][3] = kernel_access
            del kernel
            del kernel_access
            layer_lst[idx].set_weights(all_weights[idx])      

"""    
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