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
layer_cnt = 3

#"""
model_run = mr.ModelRun(db)

model_run.run_model("standard",epochs=epoch_cnt,
                    num_layers=layer_cnt,num_runs=run_cnt)

min_pruning = 10
max_pruning = 90
pruning_step = 20
for pct in range(min_pruning,max_pruning+10,pruning_step):
    model_run.run_model("sparse",epochs=epoch_cnt, num_layers=3, 
                        num_runs=run_cnt, pruning_type="weights",
                        pruning_pct=pct, pruning_change=0,
                        pruning_stage=500)

# increasing pruning
min_pruning = 10
max_pruning = 90
pruning_chg = (max_pruning-min_pruning)/epoch_cnt
model_run.run_model("sparse",epochs=epoch_cnt, num_layers=3, 
                    num_runs=run_cnt, pruning_type="weights",
                    pruning_pct=10, pruning_change=pruning_chg,
                    pruning_stage=500)

#decreasing pruning
min_pruning = 90
max_pruning = 10
pruning_chg = (max_pruning-min_pruning)/epoch_cnt
model_run.run_model("sparse",epochs=epoch_cnt, num_layers=3, 
                    num_runs=run_cnt, pruning_type="weights",
                    pruning_pct=10, pruning_change=pruning_chg,
                    pruning_stage=500)

model_run.write_to_file(filename = "TrainingResults.xls")
#"""
