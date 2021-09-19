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
    def __init__(self):
        pass
            
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
        
    def sparsify_weights(self, model, pruning_pct):
        myList = []
        sorted_lst = []
        total_len = 0
        all_weights = []
        layer_cnt = 0
        neuron_len = []
        layer_list = []
        for layer in model.layers:
            if isinstance(layer,cl.MyDense) == True:
                weights = layer.get_weights()
                all_weights.append(weights)
                neuron_len = weights[2].shape[0]
                total_len += weights[0].shape[0]
                layer_list.append(layer)
                myList.extend([(layer_cnt,layer,i,weights[2][i]) for i in range(neuron_len)])
                layer_cnt += 1
        sorted_lst = sorted(myList,key=lambda x: (x[3]))
        bottom_n = tf.cast(total_len * (pruning_pct/100),dtype=tf.int32)
        sorted_arr = np.array(sorted_lst)
        total_len = 0
        for idx in range(len(sorted_arr)):
            # initialize the weights to be removed to 0
            layer_idx = sorted_arr[idx][0]
            col_length = all_weights[layer_idx][0].shape[0]
            total_len += col_length
            if total_len < bottom_n:
                all_weights[layer_idx][0][:,sorted_arr[idx][2]] = 0
            else:
                all_weights[layer_idx][0][:col_length,sorted_arr[idx][2]] = 0
                break
            
        for idx in range(len(layer_list)):
            all_weights[idx][2][:] = 0 #sorted_arr[start:end,3]
            layer_list[idx].set_weights(all_weights[idx])
            
        