#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 18:03:11 2021

@author: tushar
"""
import tensorflow as tf
from tensorflow import keras
from keras import models
import numpy as np
import pandas as pd

import utils

class PruneLayer:
    def __init__(self, model, prune_pct):
        self.model = model
        self.layer_indices = self.__get_layer_indices()
        self.prune_pct = prune_pct
        
        # count of total trainable variables
        self.trainable_wts_detail = self.get_prune_summary()
        # stores how many weights have been pruned
        self.pruned_wts = 0
        # signifies the layer which is to be pruned
        self.current_layer = 0
        # this is the way the neuron will be updated, although we now 
        # only use the ctr mode
        self.neuron_update_type = "ctr"
        
    def __del__(self):
        del self.model
        del self.layer_indices
        del self.prune_pct
        del self.trainable_wts_detail
        del self.pruned_wts
        del self.current_layer
        del self.neuron_update_type
        
    def __get_layer_indices(self):
        layer_indices = []
        current_layer_idx = -1
        
        for layer in self.model.layers:
            current_layer_idx += 1
            if  not isinstance(layer,keras.layers.Dense) or layer.name == "output":
                continue    
            layer_indices.append(current_layer_idx)
        return layer_indices
    
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
        
        prune_pct = (pruned_wts_cnt/trainable_wts_cnt)*100
        return (trainable_wts_cnt,pruned_wts_cnt,prune_pct)
    
    def get_prune_pct(self):
        return (self.pruned_wts/self.trainable_wts_detail[0])*100
        
    def get_zeros(self):
        wts = self.model.trainable_variables
        num_zeros = 0
        for w in wts:
            if "kernel" in w.name:
                kernel = w.numpy()
                num_zeros += kernel[np.where(kernel == 0)].size
        return num_zeros

    def __update_neuron_frequency(self,data):
        output_layer = self.model.layers[self.layer_indices[self.current_layer]]
        activation_model = models.Model(inputs=self.model.input, 
                                        outputs=output_layer.output)
        activation_data = activation_model(data)
    
        condition = tf.equal(activation_data, 0)
        int_tensor = tf.cast(tf.where(condition, 1, 0), tf.int32)
        neuron_freq = tf.reduce_sum(int_tensor,axis=0)
        return neuron_freq
    
    def enable_pruning(self):
        self.model.enable_pruning()
         
    # todo: add logic to remove already zero wts 
    def prune(self, pruning_type, data):
        layer = self.model.layers[self.layer_indices[self.current_layer]]
        
        weights = layer.get_weights()
        neuron_freq = self.__update_neuron_frequency(data)
        
        kernel = weights[0]
        rows,cols = kernel.shape
        
        if pruning_type == "neuron":
            prod = neuron_freq
            
            # Enumerate the vector to get (index, value) pairs
            indexed_vector = list(enumerate(prod))
            
            # Sort the indexed vector based on values (second element of each tuple)
            sorted_vector = sorted(indexed_vector, key=lambda x: x[1])
    
            # Extract the sorted indexes
            sorted_indexes = [index for index, _ in sorted_vector]
            
            idx = tf.cast(len(sorted_indexes) * (self.prune_pct/100),dtype=tf.int32)
            
            # this will be a constant data for neuron based pruning
            pruned_wts = kernel.shape[0]
            
            for i in range(0,idx):
                prune_idx = sorted_indexes[i]
                kernel[:,prune_idx] = 0
                self.pruned_wts += pruned_wts
        else:
            # we consider the absolute value of the weights
            prod = np.abs(kernel*neuron_freq.numpy())
            total_len = prod.shape[0] * prod.shape[1]
            idx = int(total_len *self.prune_pct/100)
            
            flatten_prod = prod.flatten()
            indices = np.argsort(flatten_prod)[:idx]

            flatten_wts = kernel.flatten()
            # Set the k lowest values to 0 in the flattened matrix
            flatten_wts[indices] = 0
            
            # Reshape the flattened matrix back to its original shape
            kernel = flatten_wts.reshape(kernel.shape)
            
            self.pruned_wts += len(indices)
             
        weights[0] = kernel
        layer.set_weights(weights)
        self.current_layer += 1
        
        
class PruneNetwork:
    def __init__(self, model):
        self.model = model
        self.total_pruned_wts = 0
        self.functors = []
        self.activation_model = []
        self.__create_functors()
        #self.neuron_len = 0
        self.layer_name_lst,self.activation_lst = self.__create_activation_list()
        
        # variable for prune_optimal
        self.pruning_rate = 1
        
        # variable for cip
        self.neuron_update_type = "ctr"
        self.neuron_update = False
        
    def __del__(self):
        del self.model
        del self.functors
        del self.activation_lst
        del self.pruning_rate
        del self.neuron_update_type
    
    def __create_functors(self):
        layer_outputs = []
        for layer in self.model.layers:
            if  not isinstance(layer,keras.layers.Dense) and \
                not isinstance(layer,keras.layers.Conv2D):
                continue
            #tf.print("layer name:",layer.name)
            if layer.name == "output":
                continue
            
            layer_outputs += [layer.output]
        
        #layer_outputs = [layer.output for layer in self.model.layers] 
        self.activation_model = models.Model(inputs=self.model.input, 
                                             outputs=layer_outputs)
    """
    def __create_functors(self):
        inp = self.model.input
        outputs = []
        #neuron_lst =[]
        for layer in self.model.layers:
            if  not isinstance(layer,keras.layers.Dense) and \
                not isinstance(layer,keras.layers.Conv2D):
                continue
            #tf.print("layer name:",layer.name)
            if layer.name == "output":
                continue
            
            outputs += [layer.output]
            
            #tf_neuron_shape = layer.output.shape[1:] 
            #zeros = tf.zeros(shape=tf_neuron_shape)
            #tf_neuron_var = tf.Variable(zeros,dtype=tf.float32)
            #neuron_lst.append(tf_neuron_var)
            
        # evaluation functions
        self.functors = [K.function([inp], [out]) for out in outputs]
        #self.activation_lst = neuron_lst
    """
    
    """
    def __create_neurons(self):
        trainable_vars = self.model.trainable_variables
        neuron = []
    
        for idx in range(len(trainable_vars)):
            if "kernel" not in trainable_vars[idx].name:
                continue
            if trainable_vars[idx].name == "output":
                continue
            #rows,cols = trainable_vars[idx].shape
            #neuron.append(tf.Variable([0 for x in range(cols)], dtype=tf.float32))
            zeros = tf.zeros(trainable_vars[idx].shape,dtype=tf.float32)
            neuron.append(zeros)
            
        return neuron
    """
    def __create_activation_list(self):
        activation_lst = []
        layer_name_lst = []
        zeros = []
        layers = self.model.layers
        total_layers = len(layers)
        for idx in range(total_layers):
            if "dense" in layers[idx].name or "conv" in layers[idx].name:
                zeros = tf.zeros(layers[idx].output.shape[-1],dtype=tf.int32)
                name = layers[idx].name
                activation_lst.append(zeros)
                layer_name_lst.append(name)
        return [layer_name_lst,activation_lst]
    
    def enable_neuron_update(self, neuron_update_type):
        self.neuron_update = True
        self.neuron_update_type = neuron_update_type
    
    def disable_neuron_update(self):
        self.neuron_update = False
    
    def neuron_update_flag(self):
        return self.neuron_update
    
    def enable_pruning(self):
        self.model.enable_pruning()
    
    def update_neuron_frequency(self,data):
        idx = 0
        activations = self.activation_model(data)
        for activation_data in activations:
            #self.__update_neuron(idx,activation_data)
            act_dim = activation_data.shape
            if tf.size(act_dim) == 4: # conv layer
                activation_sum = tf.reduce_sum(activation_data, [0,1,2])
                #activation_mean = tf.reduce_mean(activation_data, axis=[0,1,2])
                self.activation_lst[idx] = tf.add(self.activation_lst[idx], activation_sum)
            else:
                #activation_sum = tf.reduce_sum(activation_data, [0])
                #activation_mean = tf.reduce_mean(activation_data, axis=[0])
                condition = tf.equal(activation_data, 0)
                int_tensor = tf.cast(tf.where(condition, 1, 0), tf.int32)
                neuron_freq = tf.reduce_sum(int_tensor,axis=0)
                self.activation_lst[idx] = tf.add(self.activation_lst[idx],neuron_freq)
                #res_1 = tf.add(self.activation_lst[idx],activation_sum)
                #self.activation_lst[idx] = tf.where(condition, self.activation_lst[idx],res_1)
            idx += 1

    def prune_cnn(self, pruning_pct, pruning_type):
        all_weights = []
        layer_lst = []
        data_lst = []
        dim_lst = []
        layer_idx = 0
        act_idx = 0
        total_len = 0
        main_idx = 0
        
        for layer in self.model.layers:
            if  not isinstance(layer,keras.layers.Dense) and \
                not isinstance(layer,keras.layers.Conv2D) or layer.name == "output":
                layer_idx += 1
                continue
            weights = layer.get_weights()
            all_weights.append(weights)
            
            # get kernel weights
            kernel = weights[0]
            dim_lst.append([kernel.shape,tf.size(kernel).numpy()])
            flatten_wts = kernel.flatten()
            
            act = self.activation_lst[act_idx].numpy()
            prod = abs(kernel * act)
            flatten_prod = prod.flatten()
            dim = len(flatten_prod)
            data_lst.extend([(layer_idx, main_idx+idx, 
                              flatten_wts[idx], flatten_prod[idx])
                             for idx in range(dim)])
            
            layer_lst.append(layer)
            total_len += dim
            main_idx = dim
            layer_idx += 1
            act_idx += 1
            del kernel
         
        df_data = pd.DataFrame(data_lst,
                               columns=["layer","index","wts","prod"])
        del data_lst
        
        df_wts = df_data.loc[df_data["wts"] != 0]
        df_zero_wt = df_data.loc[df_data["wts"] == 0]
        
        df_wts_sort = df_wts.sort_values(["prod"], ascending=True)
        
        pruning_idx = tf.cast(total_len * (pruning_pct/100),dtype=tf.int32)
        pruning_idx = pruning_idx.numpy()
        df_wts_sort.iloc[0:pruning_idx,2] = 0
        
        # also check the condition number largest sig val / smallest sig val
        
        #df_wts = df_wts_sort.append(df_zero_wt, ignore_index=True)
        df_wts = pd.concat([df_wts_sort,df_zero_wt], ignore_index=True)
        del df_wts_sort
        
        df_data = df_wts.sort_values(["layer","index"], ascending=True)
        
        # retrieve all kernel values from the sorted list
        kernel_wts = df_data["wts"]
        del df_data
        # convert the lists into arrays
        kernel_arr = np.array(kernel_wts)
        del kernel_wts
        
        start_offset = 0
        for idx in range(len(layer_lst)):
            dim = dim_lst[idx][0]
            total_elements = dim_lst[idx][1]
            end_offset = start_offset+total_elements
            kernel = np.reshape(kernel_arr[start_offset:end_offset],(dim))
            start_offset = end_offset
            all_weights[idx][0] = kernel
            del kernel
            layer_lst[idx].set_weights(all_weights[idx])                      

        
    def cnn(self, pruning_pct, pruning_type):
        all_weights = []
        layer_lst = []
        data_lst = []
        dim_lst = []
        layer_idx = 0
        total_len = 0
        
        for layer in self.model.layers:
            if  not isinstance(layer,keras.layers.Dense) and \
                not isinstance(layer,keras.layers.Conv2D):
                continue
            weights = layer.get_weights()
            all_weights.append(weights)
            
            # get kernel weights
            kernel = weights[0]
            total_dim = 1
            for dim in kernel.shape:
                total_dim *= dim
            total_len += total_dim
            dim_lst.append([kernel.shape,total_dim])
            
            if layer.name != "output":
                neuron_freq = self.activation_lst[layer_idx].numpy()
                flatten_neuron = neuron_freq.flatten()
                abs_flatten_neuron = np.abs(flatten_neuron)
                flatten_wts = kernel.flatten()
                layer_lst.append(layer)
            
                data_lst.extend([(layer_idx, i, abs_flatten_neuron[i], flatten_wts[i]) 
                           for i in range(total_dim)])
                del neuron_freq
                layer_idx += 1
            del kernel
         
        df_data = pd.DataFrame(data_lst,
                               columns=["layer","idx","flatten_neuron","flatten_wts"])
        del data_lst
        
        df_wts = df_data.loc[df_data["flatten_wts"] != 0]
        df_zero_wt = df_data.loc[df_data["flatten_wts"] == 0]
        
        df_wts_sort = df_wts.sort_values(["flatten_neuron"], ascending=True)
        
        pruning_idx = tf.cast(total_len * (pruning_pct/100),dtype=tf.int32)
        pruning_idx = pruning_idx.numpy()
        
        start_idx = 0    
        df_wts_sort.iloc[start_idx:start_idx+pruning_idx,3] = 0
        
        # also check the condition number largest sig val / smallest sig val
        
        #df_wts = df_wts_sort.append(df_zero_wt, ignore_index=True)
        df_wts = pd.concat([df_wts_sort,df_zero_wt], ignore_index=True)
        del df_wts_sort
        
        df_data = df_wts.sort_values(["layer","idx"], ascending=True)
        
        # retrieve all kernel values from the sorted list
        kernel_wts = df_data["flatten_wts"]
        del df_data
        # convert the lists into arrays
        kernel_arr = np.array(kernel_wts)
        del kernel_wts
        
        start_offset = 0
        for idx in range(len(layer_lst)):
            dim = dim_lst[idx][0]
            total_elements = dim_lst[idx][1]
            end_offset = start_offset+total_elements
            kernel = np.reshape(kernel_arr[start_offset:end_offset],(dim))
            start_offset = end_offset
            all_weights[idx][0] = kernel
            del kernel
            layer_lst[idx].set_weights(all_weights[idx])                      
    
            
    def prune_redundant_wts(self, df_data, df_data_sort, dim_lst):
        # retieve all neurons with all zero wts
        df_zero = df_data_sort.loc[df_data_sort["wts"] == 0]
        # retrieve unique columns of j
        j_values = df_zero["j"].unique()
        layers = len(dim_lst)
        
        for l in range(layers):
            for j in j_values:
                incoming_zeros = len(df_zero.loc[(df_zero["j"]==j) & (df_zero["layer"]==l) ].index)
                if dim_lst[l][0] != incoming_zeros:
                    continue
                next_layer = l + 1
                df_data.loc[(df_data["layer"] == next_layer) & (df_data["i"] == j),"wts"]=0
        return df_data
        
    def cip(self, pruning_pct, pruning_type):
        all_weights = []
        layer_lst = []
        data_lst = []
        dim_lst = []
        layer_idx = 0
        total_len = 0
        total_pruning = 0
        
        for layer in self.model.layers:
            if not isinstance(layer,keras.layers.Dense):
                continue
            weights = layer.get_weights()
            all_weights.append(weights)
            
            # get kernel weights
            kernel = weights[0]
            rows,cols = kernel.shape 
            total_len += rows*cols
            dim_lst.append([rows,cols])
            
            # get the neuron freq 
            prune_flag = 0
            prod = 0
            if layer.name != "output":
                neuron_freq = self.activation_lst[layer_idx].numpy()
                layer_lst.append(layer)
            
                data_lst.extend([(layer_idx,i,j, neuron_freq[j], kernel[i][j], prune_flag, prod) 
                           for j in range(cols) for i in range(rows)])
                del neuron_freq
                layer_idx += 1
            del kernel
         
        df_data = pd.DataFrame(data_lst,
                               columns=["layer","i","j","neuron_freq","wts","flag","prod"])
        del data_lst
        
        df_data_sort = df_data.loc[df_data["wts"] != 0]
        df_zero_wt = df_data.loc[df_data["wts"] == 0]
        
        prod = df_data_sort["neuron_freq"]
        if pruning_type == "neuron_wts":
            prod = prod * abs(df_data_sort["wts"])
        
        df_data_sort = df_data_sort.assign(prod=prod)
        del prod
        df_data_sort = df_data_sort.sort_values(["prod"], ascending=True)
        
        pruning_idx = tf.cast(total_len * (pruning_pct/100),dtype=tf.int32)
        pruning_idx = pruning_idx.numpy()
        
        start_idx = 0    
        df_data_sort.iloc[start_idx:start_idx+pruning_idx,4] = 0

        #self.redistribute_wts(df_data, layer_cnt)
        
        # also check the condition number largest sig val / smallest sig val
        
        #df_data = df_data_sort.append(df_zero_wt, ignore_index=True)
        df_data = pd.concat([df_data_sort,df_zero_wt], ignore_index=True)
        
        
        
        df_data = df_data.sort_values(["layer","i","j"], ascending=True)
        df_data_sort = df_data_sort.sort_values(["layer","i","j"], ascending=True)
        
        df_data = self.prune_redundant_wts(df_data,df_data_sort, dim_lst)
        del df_data_sort

        
        # retrieve all kernel values from the sorted list
        kernel_wts = df_data["wts"]
        del df_data
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
    
    def redistribute_wts(self, df_data, layer_cnt):
        
        # redistribute weights to remaining neurons weighted by
        # neuron frequency
        
        for layer_id in range(layer_cnt):
            df_layer_data = df_data.loc[df_data['layer'] == layer_id                                        ]
            # divide the data into two groups based on which weight is to be pruned
            df_rem = df_layer_data.loc[df_layer_data["flag"] == 0]
            df_to_prune = df_layer_data.loc[df_layer_data["flag"] == 1]
            #retrieve the weight values for the neuron to be redistributed
            while df_to_prune.empty != True:
                # get the first row data
                layer = df_to_prune.iloc[0]["layer"]
                i = df_to_prune.iloc[0]["i"]
                df_same = df_to_prune.loc[ (df_to_prune["layer"] == layer) & 
                                (df_to_prune["i"] == i)]
                wt_avg = (df_same["neuron_freq"]*df_same["wts"]).sum()/df_same["neuron_freq"].sum()
                df_rem_same = df_rem.loc[ (df_rem["layer"] == layer) & (df_rem["i"] == i)]
                neuron_freq_sum = df_rem_same["neuron_freq"].sum()
                df_rem_same.wts = df_rem_same["wts"] + (df_rem_same["neuron_freq"]/neuron_freq_sum) * wt_avg
                df_to_prune.drop(df_same.index, inplace=True)
                # update the original dataframe
                df_rem.update(df_rem_same)
            df_data.update(df_rem)
            
 
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
                neuron_freq = self.activation_lst[layer_cnt].numpy()
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
            if layer.name == "output":
                continue
            weights = layer.get_weights()
            all_weights.append(weights)
            
            # get the neuron freq 
            neuron_freq = self.activation_lst[layer_cnt].numpy()
            
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
            neuron_freq = self.activation_lst[layer_cnt].numpy()
            
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

    
    def __update_neuron(self, idx, activation_data):
        #trainable_var = self.model.trainable_variables[2*idx]
        #activation_mean = tf.math.reduce_mean(tf.abs(activation_data),axis=0)
        #activation_mean = tf.cast(activation_mean,dtype=tf.float64)
        """
        if "conv" in trainable_var.name:
            dims = activation_mean.shape
            activation_mean = tf.reshape(activation_mean, [dims[0],dims[1],1,dims[2]])
            output_neuron = tf.nn.conv2d(trainable_var, activation_mean, 
                                         strides=[1, 1, 1, 1], padding='SAME')
            self.activation_lst[idx] = tf.add(self.activation_lst[idx], output_neuron)
        
        if "conv" in trainable_var.name:
            dims = activation_mean.shape
            n_filters = dims[-1]
            ip_shape = trainable_var.shape
            ip_kernel_dim = [ip_shape[0],ip_shape[1]]
            in_ch = ip_shape[2]
            out_ch = ip_shape[3]
            output_arr = np.empty(ip_shape, dtype=float)
            for i_ch in range(in_ch):
                for o_ch in range(out_ch):
                    ip_kernel = trainable_var[:,:,i_ch,o_ch]
                    ip_kernel_dim = ip_kernel.shape
                    ip_kernel = tf.reshape(ip_kernel, [1,ip_kernel_dim[0],
                                                       ip_kernel_dim[1],1])
                    act_filter = activation_mean[:,:,o_ch]
                    act_filter_dim = act_filter.shape
                    act_filter = tf.reshape(act_filter, [act_filter_dim[0],
                                                         act_filter_dim[1],1,1])
                    
                    output = tf.nn.conv2d(ip_kernel, act_filter, 
                                          strides=[1, 1, 1, 1], padding='SAME')
                    output = output*ip_kernel
                    output_arr[:,:,i_ch,o_ch] = output[0,:,:,0]
            output_tensor = tf.convert_to_tensor(output_arr,dtype=tf.float32)
            self.activation_lst[idx] = tf.add(self.activation_lst[idx], output_tensor)
        
        self.activation_lst[idx] = tf.add(self.activation_lst[idx], activation_mean)
        """        
    
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
        
        #tf.print("Trainable variables:",trainable_wts_cnt)
        #tf.print("Variables pruned:",pruned_wts_cnt)
        prune_pct = (pruned_wts_cnt/trainable_wts_cnt)*100
        #tf.print("Prune percentage:",prune_pct)
        return (trainable_wts_cnt,pruned_wts_cnt,prune_pct)
    
    def reset_neuron_count(self):
        for idx in range(len(self.activation_lst)):
            zeros = tf.zeros(self.activation_lst[idx].shape,dtype=tf.float32)
            self.activation_lst[idx] = zeros #[0] * len(self.activation_lst[idx])
    
