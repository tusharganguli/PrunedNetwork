#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:54:36 2021

@author: tushar
"""
from tensorflow import keras

import model_run as mr


db = keras.datasets.fashion_mnist
epoch_cnt = 500
run_cnt = 1
layer_cnt = 3
# number of epochs after which pruning occurs
prune_freq = 15

tensorboard_dir = "results"
prune_dir = "prune_details"
model = mr.ModelRun(db,tensorboard_dir, prune_dir)
n_cnt = [300,100,50]
t_acc = 0.98
"""
model.evaluate(run_type="standard",epochs=epoch_cnt, num_layers=layer_cnt, 
               neuron_cnt = n_cnt, num_runs=run_cnt, pruning_type="none",
               training_accuracy = t_acc)
"""
"""
epochs: Number of epochs to trains the model
num_layers : number of layers to be added to the model
neuron_cnt : number of neurons in the respective layers of the model
num_runs: total number of times the model will run to generate an average result
pruning_type : different types of pruning to be carried out
training_accuracy : Final training accuracy to achieve
pruning_pct : Percent of remaining weights to be pruned
pruning_change : delta to add to the percent of pruning weights to be 
                pruned in the next iteration
prune_accuracy_threshold : Pruning to be started after reaching this training accuracy
prune_freq : Once accuracy is achieved, how many epochs later will the pruning start
neuron_ctr_start_at_acc : Neuron frequency update to start after reaching this training accuracy
restart_neuron_count : Reset the neuron counter after pruning
"""
model.evaluate(run_type="prune", epochs=epoch_cnt, 
               num_layers=layer_cnt, neuron_cnt = n_cnt,
               num_runs=run_cnt, pruning_type="weights",
               training_accuracy = t_acc,
               pruning_pct=5, pruning_change=0, 
               prune_accuracy_threshold=90/100,
               prune_freq=prune_freq,
               neuron_ctr_start_at_acc = 60/100,
               reset_neuron_count = True)
"""
model.eval(run_type="prune", epochs=epoch_cnt, 
               num_layers=layer_cnt, neuron_cnt = n_cnt,
               num_runs=run_cnt, pruning_type="absolute_weights",
               final_training_accuracy = t_acc,
               pruning_pct=80,
               prune_at_accuracy=90/100)
"""
"""
for p_pct in range(5,30,5):
    for acc_th in range(95,75,-5):
        model.evaluate("sparse",epochs=epoch_cnt, num_layers=layer_cnt, 
                       neuron_cnt = n_cnt, num_runs=run_cnt, pruning_type="weights",
                       training_accuracy = t_acc,
                       pruning_pct=p_pct, pruning_change=0, 
                       prune_accuracy_threshold=acc_th/100,
                       prune_freq=prune_freq)
"""      
model.write_to_file(filename = "TrainingResults.xls")

"""
model = mr.ModelRun(db,"results")


model.evaluate("standard",epochs=epoch_cnt, num_layers=layer_cnt,
               num_runs=run_cnt, pruning_type="none")

start_pruning_pct = 10
end_pruning_pct = 30
pruning_pct_step = 5

start_prune_accuracy = 65
end_prune_accuracy = 100
prune_accuracy_step = 10


for pct in range(start_pruning_pct,end_pruning_pct,pruning_pct_step):
        for prune_acc in range(start_prune_accuracy,end_prune_accuracy,prune_accuracy_step):
            model.evaluate("sparse",epochs=epoch_cnt, num_layers=layer_cnt, 
                           num_runs=run_cnt, pruning_type="weights",
                           pruning_pct=pct, pruning_change=0,
                           prune_accuracy_threshold=prune_acc/100,
                           prune_freq=prune_freq)

model.write_to_file(filename = "TrainingResults1.xls")

start_pruning_pct = 0
pruning_chg = 5

for prune_acc in range(start_prune_accuracy,end_prune_accuracy,prune_accuracy_step):
    model.evaluate("sparse",epochs=epoch_cnt, num_layers=layer_cnt, 
                   num_runs=run_cnt, pruning_type="weights",
                   pruning_pct=start_pruning_pct, 
                   pruning_change=pruning_chg,
                   prune_accuracy_threshold=prune_acc/100,
                   prune_freq=prune_freq)

model.write_to_file(filename = "TrainingResults2.xls")

start_pruning_pct = 30
pruning_chg = -5

for prune_acc in range(start_prune_accuracy,end_prune_accuracy,prune_accuracy_step):
    model.evaluate("sparse",epochs=epoch_cnt, num_layers=layer_cnt, 
                   num_runs=run_cnt, pruning_type="weights",
                   pruning_pct=start_pruning_pct, 
                   pruning_change=pruning_chg,
                   prune_accuracy_threshold=prune_acc/100,
                   prune_freq=prune_freq)

model.write_to_file(filename = "TrainingResults3.xls")

"""

