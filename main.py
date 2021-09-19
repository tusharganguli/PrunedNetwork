#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:54:36 2021

@author: tushar
"""

import model_run as mr

model_run = mr.ModelRun()

epoch_cnt = 5
run_cnt = 2
layer_cnt = 3

model_run.run_model("standard",epochs=epoch_cnt,
                    num_layers=layer_cnt,num_runs=run_cnt)

min_pruning = 10
max_pruning = 80
pruning_step = 10
#for pct in range(min_pruning,max_pruning,pruning_step):
#    model_run.run_model("sparse",epochs=epoch_cnt, num_layers=layer_cnt, 
#                        num_runs=run_cnt, pruning_pct=pct, pruning_stage=500)

#model_run.write_to_file(filename = "TrainingResults.xls")