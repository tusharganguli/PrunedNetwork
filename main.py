#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:54:36 2021

@author: tushar
"""

import model_run as mr

model_run = mr.ModelRun()
start_epoch=5
end_epoch=6
inc = 1

for num_layer in range(1,2):
    for epoch in range(start_epoch,end_epoch,inc):    
            model_run.run_model("standard",epochs=epoch,
                                num_layers=num_layer,num_runs=1)
            model_run.run_model("sparse",epochs=epoch, num_layers=num_layer, 
                                num_runs=1, pruning_pct=20, pruning_stage=500)
