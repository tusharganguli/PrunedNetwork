#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 03:26:01 2021

@author: tushar
"""

import tensorboard as tb
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import numpy as np
import math
import pandas as pd

class GeneratePlots():
    def __init__(self, experiment_id):
        self.experiment_id = experiment_id
        experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
        self.df = experiment.get_scalars()
        
    def __GetData(self, filter_value):
        data = self.df[self.df.run.str.contains(filter_value)]
        return data
    
    def __FilterData(self,data, col_name,filter_value):
        new_data = data[data[col_name].str.contains(filter_value)]
        return new_data
    
    def __ExcludeData(self,data, col_name,filter_value):
        new_data = data[~data[col_name].str.contains(filter_value)]
        return new_data
    
    def __PlotData(self,data, hue, title):
        plt.figure(figsize=(4.0, 2.8), dpi=600)
        #plt.subplot(1, 2, 1)
        plt.grid()
        #plt.rcParams.update({'font.family':'sans-serif'})
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams.update({'font.size': 8})
        ax = sns.lineplot(data=data, x="step", y="value", hue=hue, 
                          alpha=1, linewidth=0.8, ci=None)
        plt.legend(loc='lower right')
        x_ticker = range(0,data.step.max(),10)
        ax.set_xticks(x_ticker)
        #y_ticker = np.around(np.linspace(0.7,1,30),decimals=2)
        start = math.floor(data.value.min()*100)
        y_ticker = [ x/100 for x in range(start,102,2)]
        ax.set_yticks(y_ticker)
        ax.set_title(title)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Epoch")
        
    def __SavePlot(self, filename):
        #plt.savefig(filename, format='eps')
        plt.savefig(filename)
        plt.close()
        
    def ConvertToEps(self, prune_dir):
        import glob, os
        os.chdir("./" + prune_dir)
        for file in glob.glob("*.pdf"):
            command = "inkscape " + file + " -o " + file.split(".")[0] + ".eps"
            os.system(command)
            
    def PlotOptimal(self, prune_dir):
        
        self.__PlotResetNeuron(prune_dir, "EpochInterval_2", "TotalPruning_2")
        self.__PlotResetNeuron(prune_dir, "EpochInterval_2", "TotalPruning_4")
        self.__PlotResetNeuron(prune_dir, "EpochInterval_2", "TotalPruning_6")
        self.__PlotResetNeuron(prune_dir, "EpochInterval_2", "TotalPruning_8")
        self.__PlotResetNeuron(prune_dir, "EpochInterval_2", "TotalPruning_10")
        
        self.__PlotResetNeuron(prune_dir, "EpochInterval_10", "TotalPruning_2")
        self.__PlotResetNeuron(prune_dir, "EpochInterval_10", "TotalPruning_4")
        self.__PlotResetNeuron(prune_dir, "EpochInterval_10", "TotalPruning_6")
        self.__PlotResetNeuron(prune_dir, "EpochInterval_10", "TotalPruning_8")
        self.__PlotResetNeuron(prune_dir, "EpochInterval_10", "TotalPruning_10")
        
        self.__PlotResetNeuron(prune_dir, "EpochInterval_20", "TotalPruning_2")
        self.__PlotResetNeuron(prune_dir, "EpochInterval_20", "TotalPruning_4")
        self.__PlotResetNeuron(prune_dir, "EpochInterval_20", "TotalPruning_6")
        self.__PlotResetNeuron(prune_dir, "EpochInterval_20", "TotalPruning_8")
        self.__PlotResetNeuron(prune_dir, "EpochInterval_20", "TotalPruning_10")
        
        self.__PlotResetNeuron(prune_dir, "EpochInterval_40", "TotalPruning_3")
        
        
        self.__PlotNumberOfPruning(prune_dir, "TotalPruning_2", reset_neuron = True)
        self.__PlotNumberOfPruning(prune_dir, "TotalPruning_4", reset_neuron = True)
        self.__PlotNumberOfPruning(prune_dir, "TotalPruning_6", reset_neuron = True)
        self.__PlotNumberOfPruning(prune_dir, "TotalPruning_8", reset_neuron = True)
        self.__PlotNumberOfPruning(prune_dir, "TotalPruning_10", reset_neuron = True)
        
        self.__PlotNumberOfPruning(prune_dir, "TotalPruning_2")
        self.__PlotNumberOfPruning(prune_dir, "TotalPruning_4")
        self.__PlotNumberOfPruning(prune_dir, "TotalPruning_6")
        self.__PlotNumberOfPruning(prune_dir, "TotalPruning_8")
        self.__PlotNumberOfPruning(prune_dir, "TotalPruning_10")
        
        self.__PlotEpochInterval(prune_dir, "EpochInterval_2", reset_neuron = True)
        self.__PlotEpochInterval(prune_dir, "EpochInterval_10", reset_neuron = True)
        self.__PlotEpochInterval(prune_dir, "EpochInterval_20", reset_neuron = True)
        self.__PlotEpochInterval(prune_dir, "EpochInterval_40", reset_neuron = True)
        
        self.__PlotEpochInterval(prune_dir, "EpochInterval_2")
        self.__PlotEpochInterval(prune_dir, "EpochInterval_10")
        self.__PlotEpochInterval(prune_dir, "EpochInterval_20")
        self.__PlotEpochInterval(prune_dir, "EpochInterval_40")
        
        
    
    def __PlotResetNeuron(self, prune_dir, interval_type, prune_type):
        data = self.__GetData("train")
        data = self.__FilterData(data,"tag","accuracy")
        data = self.__FilterData(data,"run","optimal")
        data = self.__FilterData(data,"run",interval_type + "_")
        data = self.__FilterData(data,"run",prune_type + "_")
        
        split_name = data.run.apply(lambda label: label.split("/")[0])
        split_name = split_name.apply(lambda label: label.split("_"))
        hue_name = split_name.apply(lambda label: "Reset Neuron Enabled" 
                                    if label[10] == "ResetNeuron" else 
                                    "Reset Neuron Disabled")
        
        interval_label = interval_type.split("_")
        pruning_label = prune_type.split("_")
        title = "Epoch Interval:" + interval_label[1] + \
                ", No. of Pruning: " + pruning_label[1]
        
        self.__PlotData(data,hue_name, title)
        self.__SavePlot(prune_dir + "/TrainingAccuracy_ResetNeuron_" + interval_type + 
                        "_" + prune_type + ".pdf")

    def __PlotEpochInterval(self, prune_dir, run_type, reset_neuron=False):
        data = self.__GetData("train")
        data = self.__FilterData(data,"tag","accuracy")
        standard_data = self.__FilterData(data,"run", "standard")
        std_hue_name = standard_data.run.apply(lambda label: "standard")
        if reset_neuron == True:
            data = self.__FilterData(data,"run","ResetNeuron")
        else:
            data = self.__ExcludeData(data,"run","ResetNeuron")
        optimal_data = self.__FilterData(data,"run","optimal")
        optimal_data = self.__FilterData(data,"run",run_type)
        split_name = optimal_data.run.apply(lambda label: label.split("/")[0])
        split_name = split_name.apply(lambda label: label.split("_"))
        hue_name = split_name.apply(lambda label: "No. of Pruning:" + label[7])
        hue_name = hue_name.append(std_hue_name)
        data = optimal_data.append(standard_data)
        label = run_type.split("_")
        title = "Epoch Interval:" + label[1] + ", Reset Neuron: "
        
        if reset_neuron == True:
            title += "Enabled"
            run_type += "_ResetNeuron_Enabled"
        else:
            title += "Disabled"
            run_type += "_ResetNeuron_Disabled"
        self.__PlotData(data,hue_name, title)
        self.__SavePlot(prune_dir + "/TrainingAccuracy_" + run_type + ".pdf")

    def __PlotNumberOfPruning(self, prune_dir, run_type, reset_neuron=False):
        data = self.__GetData("train")
        data = self.__FilterData(data,"tag","accuracy")
        standard_data = self.__FilterData(data,"run", "standard")
        std_hue_name = standard_data.run.apply(lambda label: "standard")
        if reset_neuron == True:
            data = self.__FilterData(data,"run","ResetNeuron")
        else:
            data = self.__ExcludeData(data,"run","ResetNeuron")
        optimal_data = self.__FilterData(data,"run","optimal")
        optimal_data = self.__FilterData(data,"run",run_type)
        split_name = optimal_data.run.apply(lambda label: label.split("/")[0])
        split_name = split_name.apply(lambda label: label.split("_"))
        hue_name = split_name.apply(lambda label: "Epoch Interval:" + label[5])
        hue_name = hue_name.append(std_hue_name)
        data = optimal_data.append(standard_data)
        label = run_type.split("_")
        
        title = "No. of Pruning:" + label[1] + ", Reset Neuron: "
        
        if reset_neuron == True:
            title += "Enabled"
            run_type += "_ResetNeuron_Enabled"
        else:
            title += "Disabled"
            run_type += "_ResetNeuron_Disabled"
        self.__PlotData(data,hue_name, title)
        self.__SavePlot(prune_dir + "/TrainingAccuracy_" + run_type + ".pdf")
        
    def AnalyzeLoss(self, prune_dir):
        #self.__CalculateEILoss("train", prune_dir)
        #self.__CalculateEILoss("validation", prune_dir)
        #self.__CalculateNumPruningLoss("train", prune_dir)
        #self.__CalculateNumPruningLoss("validation", prune_dir)
        self.__CalculateResetNeuronLoss("train", prune_dir)
        
    def __CalculateEILoss(self, run_type, prune_dir):
        data = self.__GetData(run_type)
        data = self.__FilterData(data,"tag","loss")
        min_loss = data.groupby("run", as_index=False).agg({"value": "min"})
        ei_2_loss = min_loss.loc[min_loss.run.str.contains("EpochInterval_2_")]
        ei_10_loss = min_loss.loc[min_loss.run.str.contains("EpochInterval_10_")]
        ei_20_loss = min_loss.loc[min_loss.run.str.contains("EpochInterval_20_")]
        ei_40_loss = min_loss.loc[min_loss.run.str.contains("EpochInterval_40_")]
        std_loss = min_loss.loc[min_loss.run.str.contains("standard")]
        min_val_loss = pd.concat([std_loss,ei_2_loss, ei_10_loss, ei_20_loss, ei_40_loss])
        run = min_val_loss.run.apply(lambda label: label.split("/")[0])
        run = run.apply(lambda label: "EI-" + label.split("_")[5] 
                        if "Epoch" in label.split("_")[4] else "Standard")
        min_val_loss['x'] = run
        bp = sns.boxplot(data=min_val_loss, y=min_val_loss.value, x=min_val_loss.x)
        bp.set(xlabel='Run',ylabel='Epoch Loss')
        self.__SavePlot(prune_dir + "/" + run_type + "_loss_EpochInterval.pdf")
        
    def __CalculateNumPruningLoss(self, run_type, prune_dir):
        data = self.__GetData(run_type)
        data = self.__FilterData(data,"tag","loss")
        min_loss = data.groupby("run", as_index=False).agg({"value": "min"})
        np_2_loss = min_loss.loc[min_loss.run.str.contains("TotalPruning_2_")]
        np_4_loss = min_loss.loc[min_loss.run.str.contains("TotalPruning_4_")]
        np_6_loss = min_loss.loc[min_loss.run.str.contains("TotalPruning_6_")]
        np_8_loss = min_loss.loc[min_loss.run.str.contains("TotalPruning_8_")]
        np_10_loss = min_loss.loc[min_loss.run.str.contains("TotalPruning_10_")]
        std_loss = min_loss.loc[min_loss.run.str.contains("standard")]
        min_val_loss = pd.concat([np_2_loss, np_4_loss,
                                  np_6_loss,np_8_loss,np_10_loss])
        run = min_val_loss.run.apply(lambda label: label.split("/")[0])
        run = run.apply(lambda label: "NP-" + label.split("_")[7] )
        std_run = std_loss.run.apply(lambda label: "Standard")
        min_val_loss = pd.concat([std_loss, min_val_loss])
        run = pd.concat([std_run,run])
        min_val_loss['x'] = run
        bp = sns.boxplot(data=min_val_loss, y="value", x=min_val_loss.x)
        bp.set(xlabel='Run',ylabel='Epoch Loss')
        self.__SavePlot(prune_dir + "/" + run_type + "_loss_NumPruning.pdf")
        
    def __CalculateResetNeuronLoss(self, run_type, prune_dir):
        data = self.__GetData(run_type)
        data = self.__FilterData(data,"tag","loss")
        min_loss = data.groupby("run", as_index=False).agg({"value": "min"})
        opt_data = self.__FilterData(min_loss,"run","optimal_")
        rn_loss = self.__FilterData(opt_data,"run","ResetNeuron_")
        no_rn_loss = self.__ExcludeData(opt_data,"run","ResetNeuron_")
        min_val_loss = pd.concat([rn_loss, no_rn_loss])
        run = min_val_loss.run.apply(lambda label: label.split("/")[0])
        run = run.apply(lambda label: "Reset Enabled" 
                        if label.split("_")[10] == "ResetNeuron" else "Reset Disabled" )
        std_loss = self.__FilterData(min_loss,"run","standard_")
        std_run = std_loss.run.apply(lambda label: "Standard")
        min_val_loss = pd.concat([std_loss, min_val_loss])
        run = pd.concat([std_run,run])
        min_val_loss['x'] = run
        bp = sns.boxplot(data=min_val_loss, y="value", x=min_val_loss.x)
        bp.set(xlabel='Run',ylabel='Epoch Loss')
        self.__SavePlot(prune_dir + "/" + run_type + "_loss_ResetNeuron.pdf")
"""
#run1_training = df[df.run.str.contains(df.iloc[0].run)]
#run1_training_acc = run1_training[run1_training.tag.str.contains("accuracy")]
training = df[df.run.str.contains("train")]
training_acc = training[training.tag.str.contains("accuracy")]
run_type = training_acc.run.apply(lambda label: label.split("/")[0])
split_name = run_type.apply(lambda label: label.split("_"))
#hue_name = split_name.apply(lambda label: label[4]+label[5]+label[6]+label[7])
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
sns.lineplot(data=training_acc, x="step", 
             y="value", hue=run_type).set_title("Training Accuracy")
plt.savefig('destination_path.eps', format='eps')
#run1_training.run.split('/')[0]
"""