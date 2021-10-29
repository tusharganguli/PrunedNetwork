#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:54:36 2021

@author: tushar
"""
from tensorflow import keras

import model_run as mr
import generate_plots as gp

db = keras.datasets.fashion_mnist
run_cnt = 1
# number of epochs after which pruning occurs
prune_freq = 15
t_acc = 0.98

prune_dir = "optimal_prune_details"
experiment_id = "VToMCmUyQPOlxHA2KAdkAQ"
plots = gp.GeneratePlots(experiment_id)
#plots.PlotOptimal(prune_dir)
#plots.ConvertToEps(prune_dir)
plots.AnalyzeLoss(prune_dir)

"""
tensorboard_dir = "standard_results"
prune_dir = "standard_details"
model = mr.ModelRun(db,tensorboard_dir, prune_dir)


model.evaluate_standard(run_type="standard", 
                        num_runs=run_cnt, 
                        final_training_accuracy = t_acc)
model.write_to_file(filename = "Results_Standard.xls")
"""

"""
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

"""
model.evaluate(run_type="prune", 
               num_runs=run_cnt, pruning_type="weights",
               final_training_accuracy = 80/100,
               pruning_pct=5, pruning_change=0, 
               prune_accuracy_threshold=70/100,
               prune_freq=2,
               neuron_update_at_acc = 65/100,
               reset_neuron_count = True)
"""

"""
tensorboard_dir = "test_results"
prune_dir = "test_prune_details"
model = mr.ModelRun(db,tensorboard_dir, prune_dir)

model.evaluate_optimal_pruning(run_type="optimal",
                               num_runs=run_cnt, 
                               pruning_type="avg_freq",
                               final_training_accuracy = 98/100,
                               epoch_pruning_interval = 2,
                               num_pruning = 4,
                               reset_neuron_count = True)
"""

"""
tensorboard_dir = "optimal_results"
prune_dir = "optimal_prune_details"
model = mr.ModelRun(db,tensorboard_dir, prune_dir)

  
n_pruning = [2,4,6,8,10]
n_intervals = [2,10,20]
reset_type = [True,False]

for r_idx in reset_type:
    for n_idx in n_intervals:
        for p_idx in n_pruning:
            model.evaluate_optimal_pruning(run_type="optimal",
                                           num_runs=run_cnt, 
                                           pruning_type="avg_freq",
                                           final_training_accuracy = 98/100,
                                           epoch_pruning_interval = n_idx,
                                           num_pruning = p_idx,
                                           reset_neuron_count = r_idx)
        
    
    
    model.evaluate_optimal_pruning(run_type="optimal",
                                       num_runs=run_cnt, 
                                       pruning_type="avg_freq",
                                       final_training_accuracy = 98/100,
                                       epoch_pruning_interval = 40,
                                       num_pruning = 3,
                                       reset_neuron_count = r_idx)

model.write_to_file(filename = "Results_OptimalPruning.xls")

"""

"""
Evaluates constant pruning at regular intervals
"""

"""
model.evaluate_interval_pruning(run_type="interval", 
                                num_runs=run_cnt, 
                                pruning_type="weights",
                                final_training_accuracy = 60/100,
                                pruning_values = [40,20,10,10],
                                epoch_range = [20,40,80,120],
                                reset_neuron_count = True)

model.write_to_file(filename = "Results_IntervalPruning.xls")
"""

"""
Evaluates one time pruning
"""
"""
model.evaluate_otp(run_type="otp", 
                   num_runs=run_cnt, 
                   pruning_type="weights",
                   neuron_update_at_acc = 35/100,
                   target_prune_pct=80,
                   prune_at_accuracy=40/100,
                   final_training_accuracy = t_acc)

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
#model.write_to_file(filename = "TrainingResults.xls")

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

