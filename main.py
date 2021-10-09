#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:54:36 2021

@author: tushar
"""
from tensorflow import keras

import model_run as mr


db = keras.datasets.fashion_mnist
epoch_cnt = 100
run_cnt = 1
layer_cnt = 3
# number of epochs after which the sparsification will occur
prune_freq = 3

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

model = mr.ModelRun(db,"results")
n_cnt = [300,100,50]
t_acc = 0.9
model.evaluate(run_type="standard",epochs=epoch_cnt, num_layers=layer_cnt, 
               neuron_cnt = n_cnt, num_runs=run_cnt, pruning_type="none",
               training_accuracy = t_acc)


model.evaluate("sparse",epochs=epoch_cnt, num_layers=layer_cnt, 
               neuron_cnt = n_cnt, num_runs=run_cnt, pruning_type="weights",
               training_accuracy = t_acc,
               pruning_pct=5, pruning_change=0, 
               prune_accuracy_threshold=80/100,
               prune_freq=prune_freq)

