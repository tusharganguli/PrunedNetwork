#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 12:01:19 2022

@author: tushar
"""

import pandas as pd
import os

class LogHandler:
    df = pd.DataFrame(columns = ['Epoch', 
                                 'Total Trainable Wts',
                                 'Total Pruned Wts',
                                 'Prune Percentage'
                                 ])
    
    def __init__(self, prune_dir, log_file_name):    
        self.prune_dir = prune_dir
        if not os.path.exists(self.prune_dir):
            os.makedirs(self.prune_dir)
        self.log_file_name = log_file_name
    
    def log(self, epoch, total_trainable_wts, total_pruned_wts, prune_pct):
        log_data = [epoch,total_trainable_wts.numpy(),
                    total_pruned_wts.numpy(),
                    prune_pct.numpy()]
        df2 = pd.DataFrame([log_data], columns=list(LogHandler.df))
        LogHandler.df = LogHandler.df.append(df2,ignore_index = True)
    
    def write_to_file(self):
        writer = pd.ExcelWriter(self.prune_dir + self.log_file_name + ".xls")
        LogHandler.df.to_excel(writer)
        # save the excel
        writer.save()
        LogHandler.df = LogHandler.df.iloc[0:0]
