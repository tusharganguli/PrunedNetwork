#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:54:36 2021

@author: tushar
"""
from tensorflow import keras

import model_run as mr


db = keras.datasets.fashion_mnist
epoch_cnt = 10
run_cnt = 1
layer_cnt = 4
# number of batch in each epoch after which neuron frequency will be updated
neuron_freq = 100
# number of epochs after which the sparsification will occur
sparse_freq = 2

model_run = mr.ModelRun(db, "custom")


model_run.run_model("sparse",epochs=epoch_cnt, num_layers=layer_cnt, 
                        num_runs=run_cnt, pruning_type="weights",
                        pruning_pct=10, pruning_change=0,
                        neuron_update_freq=neuron_freq,
                        sparse_update_freq=sparse_freq)

"""
model_run.run_model("standard",epochs=epoch_cnt,
                    num_layers=layer_cnt,num_runs=run_cnt, pruning_type="none")

min_pruning = 2
max_pruning = 10
pruning_step = 2
for pct in range(min_pruning,max_pruning+pruning_step,pruning_step):
    model_run.run_model("sparse",epochs=epoch_cnt, num_layers=layer_cnt, 
                        num_runs=run_cnt, pruning_type="neurons",
                        pruning_pct=pct, pruning_change=0,
                        neuron_update_freq=neuron_freq,
                        sparse_update_freq=sparse_freq)

min_pruning = 10
max_pruning = 90
pruning_step = 20
for pct in range(min_pruning,max_pruning+pruning_step,pruning_step):
    model_run.run_model("sparse",epochs=epoch_cnt, num_layers=layer_cnt, 
                        num_runs=run_cnt, pruning_type="weights",
                        pruning_pct=pct, pruning_change=0,
                        neuron_update_freq=neuron_freq,
                        sparse_update_freq=sparse_freq)

min_pruning = 2
max_pruning = 10
pruning_step = 2
for pct in range(min_pruning,max_pruning+pruning_step,pruning_step):
    model_run.run_model("sparse",epochs=epoch_cnt, num_layers=layer_cnt, 
                        num_runs=run_cnt, pruning_type="neuronweights",
                        pruning_pct=pct, pruning_change=0,
                        neuron_update_freq=neuron_freq,
                        sparse_update_freq=sparse_freq)

# neurons
# increasing pruning
start_pruning = 2
end_pruning = 10
pruning_chg = sparse_freq*(end_pruning-start_pruning)/epoch_cnt
model_run.run_model("sparse",epochs=epoch_cnt, num_layers=layer_cnt, 
                    num_runs=run_cnt, pruning_type="neurons",
                    pruning_pct=start_pruning, pruning_change=pruning_chg,
                    neuron_update_freq=neuron_freq,
                    sparse_update_freq=sparse_freq)

#decreasing pruning
start_pruning = 10
end_pruning = 2
pruning_chg = sparse_freq*(end_pruning-start_pruning)/epoch_cnt
model_run.run_model("sparse",epochs=epoch_cnt, num_layers=layer_cnt, 
                    num_runs=run_cnt, pruning_type="neurons",
                    pruning_pct=start_pruning, pruning_change=pruning_chg,
                    neuron_update_freq=neuron_freq,
                    sparse_update_freq=sparse_freq)

# weights
# increasing pruning
start_pruning = 10
end_pruning = 90
pruning_chg = sparse_freq*(end_pruning-start_pruning)/epoch_cnt
model_run.run_model("sparse",epochs=epoch_cnt, num_layers=layer_cnt, 
                    num_runs=run_cnt, pruning_type="weights",
                    pruning_pct=start_pruning, pruning_change=pruning_chg,
                    neuron_update_freq=neuron_freq,
                    sparse_update_freq=sparse_freq)

#decreasing pruning
start_pruning = 90
end_pruning = 10
pruning_chg = sparse_freq*(end_pruning-start_pruning)/epoch_cnt
model_run.run_model("sparse",epochs=epoch_cnt, num_layers=layer_cnt, 
                    num_runs=run_cnt, pruning_type="weights",
                    pruning_pct=start_pruning, pruning_change=pruning_chg,
                    neuron_update_freq=neuron_freq,
                    sparse_update_freq=sparse_freq)

# neuronweights
# increasing pruning
start_pruning = 2
end_pruning = 10
pruning_chg = sparse_freq*(end_pruning-start_pruning)/epoch_cnt
model_run.run_model("sparse",epochs=epoch_cnt, num_layers=layer_cnt, 
                    num_runs=run_cnt, pruning_type="neuronweights",
                    pruning_pct=start_pruning, pruning_change=pruning_chg,
                    neuron_update_freq=neuron_freq,
                    sparse_update_freq=sparse_freq)

#decreasing pruning
start_pruning = 10
end_pruning = 2
pruning_chg = sparse_freq*(end_pruning-start_pruning)/epoch_cnt
model_run.run_model("sparse",epochs=epoch_cnt, num_layers=layer_cnt, 
                    num_runs=run_cnt, pruning_type="neuronweights",
                    pruning_pct=start_pruning, pruning_change=pruning_chg,
                    neuron_update_freq=neuron_freq,
                    sparse_update_freq=sparse_freq)

model_run.write_to_file(filename = "TrainingResults.xls")
"""
