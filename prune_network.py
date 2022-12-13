#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 18:03:11 2021

@author: tushar
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from keras import models
import numpy as np
import pandas as pd

class PruneNetwork:
    def __init__(self, model):
        self.model = model
        self.total_pruned_wts = 0
        self.functors = []
        self.activation_model = []
        self.__create_functors()
        #self.neuron_len = 0
        self.neuron = self.__create_neurons()
        
        # variable for prune_optimal
        self.pruning_rate = 1
        
        # variable for cip
        self.neuron_update_type = "ctr"
     
    def __del__(self):
        del self.model
        del self.functors
        del self.neuron
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
        self.activation_model = models.Model(inputs=self.model.input, outputs=layer_outputs)
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
        #self.neuron = neuron_lst
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
    
    
    def enable_neuron_update(self, neuron_update_type):
        self.model.enable_neuron_update()
        self.neuron_update_type = neuron_update_type
    
    def disable_neuron_update(self):
        self.model.disable_neuron_update()
        
    def enable_pruning(self):
        self.model.enable_pruning()
            
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
            prod = 1
            for dim in kernel.shape:
                prod *= dim
            total_len += prod
            dim_lst.append(kernel.shape)
            
            # get the neuron freq 
            prune_flag = 0
            prod = 0
            if layer.name != "output":
                neuron_freq = self.neuron[layer_idx].numpy()
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
        
        df_data_sort = df_data_sort.assign(prod=prod)
        del prod
        df_data_sort = df_data_sort.sort_values(["prod"], ascending=True)
        
        pruning_idx = tf.cast(total_len * (pruning_pct/100),dtype=tf.int32)
        pruning_idx = pruning_idx.numpy()
        
        start_idx = 0    
        df_data_sort.iloc[start_idx:start_idx+pruning_idx,4] = 0
        
        #self.redistribute_wts(df_data, layer_cnt)
        
        # also check the condition number largest sig val / smallest sig val
        
        df_data = df_data_sort.append(df_zero_wt, ignore_index=True)
        del df_data_sort
        
        df_data = df_data.sort_values(["layer","i","j"], ascending=True)
        
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
    
    def prune_redundant_wts(self, df_data, df_data_sort, dim_lst):
        # retieve all neurons with all zero wts
        df_zero = df_data_sort.loc[df_data_sort["wts"] == 0]
        # retireve unique columns of j
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
                neuron_freq = self.neuron[layer_idx].numpy()
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

    def update_neuron_frequency(self,data, curr_acc, nw_type):
        idx = 0
        activations = self.activation_model(data)
        for activation_data in activations:
            if nw_type == "cnn":
                self.__update_cnn_neuron(idx,activation_data)
            else:
                self.__update_frequency(idx,activation_data)
            idx += 1
    """    
    def update_neuron_frequency(self,x, curr_acc, nw_type):
        idx = 0
        for func in self.functors:
            #activation_data = func(x)
            activation_data = tf.constant([[1,2]])
            if nw_type == "cnn":
                self.__update_cnn_neuron(idx,activation_data[0])
            else:
                self.__update_frequency(idx,activation_data[0])
            idx += 1

    """
    
    """
    def __update_frequency( self, idx, activation_data,curr_acc):
        
        neuron_inc = 1
        
        activation_dim = activation_data.shape
        for step in range(0,activation_dim[0]):
            
            if self.neuron_update_type == "act_acc":
                neuron_inc = curr_acc*(activation_data[step])
            elif self.neuron_update_type == "act":
                neuron_inc = activation_data[step]
                
            #condition = tf.equal(activation_data[step], 0)
            #res_1 = tf.add(self.neuron[idx],neuron_inc)
            #self.neuron[idx] = tf.where(condition, self.neuron[idx],res_1)
            res_1 = tf.cast(tf.abs(neuron_inc),dtype=tf.float32)
            self.neuron[idx] = tf.add(self.neuron[idx], res_1)
    """

    def __update_frequency( self, idx, activation_data):
        activation_dim = activation_data.shape
        act_sum = np.sum(np.abs(activation_data),axis=0)
        act_avg = act_sum/activation_dim[0]
        self.neuron[idx] = tf.add(self.neuron[idx], act_avg)
        
    def __update_cnn_neuron(self, idx, activation_data):
        filter_count = activation_data.shape[-1]
        trainable_var = self.model.trainable_variables[2*idx]
        kernel_sz = trainable_var.shape[0:2]
        
        #if "dense" in trainable_var.name:
        #    self.__update_frequency(idx,activation_data)
        
        #"""
        if "conv" in trainable_var.name:
            output_neuron = tf.keras.layers.Conv2D( filter_count, kernel_sz, 
                                   activation='relu', padding="same", 
                                   input_shape=kernel_sz )(trainable_var)
            self.neuron[idx] = tf.add(self.neuron[idx], output_neuron)
        elif "dense" in trainable_var.name:
            self.__update_frequency(idx,activation_data)
        else:
            raise Exception("Unknown layer")
        #"""
        
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
            zeros = tf.zeros(self.neuron[idx].shape,dtype=tf.float32)
            self.neuron[idx] = zeros #[0] * len(self.neuron[idx])
    
