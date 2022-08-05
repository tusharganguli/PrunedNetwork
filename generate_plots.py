#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 03:26:01 2021

@author: tushar
"""

import tensorboard as tb
from matplotlib import pyplot as plt
import seaborn as sns
import math
import pandas as pd
from tensorflow import keras
import numpy as np

def matrix_heatmap(std_dir, freq_model_dir, 
                   tf_model_dir, dir_name):
    
    std_model = keras.models.load_model(std_dir)    
    freq_model = keras.models.load_model(freq_model_dir)    
    tf_model = keras.models.load_model(tf_model_dir)
    
    plt.rc('xtick', labelsize=6)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=6)
    
    idx = 0
    num_layers = len(freq_model.layers)
    for idx in range(num_layers):
        s_layer = std_model.layers[idx]
        f_layer = freq_model.layers[idx]
        tf_layer = tf_model.layers[idx]
        
        if not isinstance(f_layer,keras.layers.Dense):
            continue
        layer_name = s_layer.name
        
        s_wts = s_layer.get_weights()[0]
        dim = s_wts.shape
        
        s_bool = s_wts > 0
        s_bool = np.invert(s_bool)
        
        f_wts = f_layer.get_weights()[0]
        f_bool = f_wts > 0
        f_bool = np.invert(f_bool)
        
        tf_wts = tf_layer.get_weights()[0]
        tf_bool = tf_wts > 0
        tf_bool = np.invert(tf_bool)
        
        #c_bool = s_bool == f_bool
        
        fig, (ax1,ax2,ax3) = plt.subplots(1,3)
        title = layer_name + ": " + str(dim[0]) + "x" + str(dim[1])
        fig.suptitle( title, fontsize=10)
        
        ax1.set_title('Standard')
        im1 = ax1.imshow(s_bool, cmap='hot', interpolation='nearest')
        im1.set_clim(0,1)
        
        ax2.set_title('Frequency')
        im2 = ax2.imshow(f_bool, cmap='hot', interpolation='nearest')
        im2.set_clim(0,1)
        
        ax3.set_title('Magnitude')
        im3 = ax3.imshow(tf_bool, cmap='hot', interpolation='nearest')
        im3.set_clim(0,1)
        
        #axs[1,1].set_title('Difference')
        #im4 = axs[1,1].imshow(c_bool, cmap='hot', interpolation='nearest')
        
        filename = dir_name + "/" + layer_name + ".png"
        plt.savefig(filename, dpi=600)
        plt.show()
        

class SVDPlots():
    def __init__(self):
        pass
    
    def ConvertToEps(self, prune_dir):
        import glob, os
        cwd = os.getcwd()
        os.chdir("./" + prune_dir)
        filetypes = ("*.pdf","*.png")
        
        for extension in filetypes:
            for file in glob.glob(extension):
                command = "inkscape " + file + " -o " + file.split(".")[0] + ".eps"
                os.system(command)
        os.chdir(cwd)
        
    def PlotSVDiff(self, sv1_df, sv2_df, num_layers):
        """
        

        Parameters
        ----------
        sv1_df : Dataframe
            Contains the singular values of each layer of the first model.
        sv2_df : Dataframe
            Contains the singular values of each layer of the second model.
        num_layers : int
            Contains the total number of layers in the model. Currently comparison 
            is across models with equal number of layers 

        Returns
        -------
        None.

        """
    
    def PlotRatio(self, svd_df, svd_plot_info, num_layers, 
                  final_svd, final_acc, prune_dir):
        
        start = 0
        end = (2*num_layers)-1
        for index, row in svd_plot_info.iterrows():
            curr_acc = row["cacc"]
            total_pruning_pct = row["tpp"]
            
            self.__PlotRelativeRatio(svd_df.loc[:,start:end], curr_acc, 
                                     total_pruning_pct, num_layers, prune_dir)
            start = end+1
            end = start+(2*num_layers)-1
        self.__PlotAbsoluteRatio(svd_df, curr_acc, total_pruning_pct, num_layers, prune_dir)
        # plots the relative difference between the svd before the start of pruning 
        # and after the final accuracy is achieved.
        #start_svd_df = svd_df.loc[:,0:5]
        #self.__PlotFinalRatio(start_svd_df, final_svd, final_acc, num_layers, prune_dir)
        
    def __PlotRelativeRatio(self, svd_df, curr_acc, total_pruning_pct, 
                            num_layers, prune_dir):
        c = ['r','g','b','y','k']
        fig, axs = plt.subplots(num_layers, 1, figsize=(8, 5), constrained_layout=True,)
       
    
        for idx,ax in enumerate(axs.flat):
            bp = svd_df.iloc[:,-(2*num_layers)+idx]
            ap = svd_df.iloc[:,-num_layers+idx]
            # process all columns to remove values < .001
            # do not set lower value for "before pruning" series as it will 
            # result in divide by 0 error
            #bp.where(bp > .001, 0, inplace=True)
            
            ap.where(ap > .001, 0, inplace=True)
            pct_change = ((ap/bp)-1)*100
            total_sig_vals = pct_change.count()
            pct_change = pct_change[pct_change != -100]
            
            ax.plot(pct_change.index, pct_change.values, color=c[idx])
            
            rem_sig_vals = pct_change.count()
            title = "Layer " + str(idx+1)
            title += ", Total Singular Values:" + str(total_sig_vals)
            title += ", Remaining Neurons:" + str(rem_sig_vals)
            ax.set_title(title, fontsize='small', loc='left')
            
        title = "Relative Ratio, Accuracy:" + str(format(curr_acc*100,".2f")) + "%"
        title += ", Total Pruning:" + str(format(total_pruning_pct,".2f")) + "%"
        fig.suptitle(title)
        fig.supxlabel("Singular Values")
        fig.supylabel("Pct. Change")
        filename = "relative_ratio_at_accuracy_" + str(format(curr_acc*100,".0f")) + ".pdf" 
        filename = prune_dir + "/" + filename
        plt.savefig(filename, dpi=600)
        plt.show()
        print("plot")
    
    def __PlotAbsoluteRatio(self, svd_df, curr_acc, total_pruning_pct, 
                         num_layers, prune_dir):
        c = ['r','g','b','y','k']
        fig, axs = plt.subplots(num_layers, 1, figsize=(8, 5), constrained_layout=True,)
       
    
        for idx,ax in enumerate(axs.flat):
            bp = svd_df.iloc[:,idx]
            ap = svd_df.iloc[:,-num_layers+idx]
            # process all columns to remove values < .001
            # do not set lower value for "before pruning" series as it will 
            # result in divide by 0 error
            #bp.where(bp > .001, 0, inplace=True)
            
            ap.where(ap > .001, 0, inplace=True)
            pct_change = ((ap/bp)-1)*100
            total_sig_vals = pct_change.count()
            pct_change = pct_change[pct_change != -100]
            
            ax.plot(pct_change.index, pct_change.values, color=c[idx])
            
            rem_sig_vals = pct_change.count()
            title = "Layer " + str(idx+1)
            title += ", Total Singular Values:" + str(total_sig_vals)
            title += ", Remaining Singular Values:" + str(rem_sig_vals)
            ax.set_title(title, fontsize='small', loc='left')
            
        title = "Absolute Ratio, Accuracy:" + str(format(curr_acc*100,".2f")) + "%"
        title += ", Total Pruning:" + str(format(total_pruning_pct,".2f")) + "%"
        fig.suptitle(title)
        fig.supxlabel("Singular Values")
        fig.supylabel("Pct. Change")
        filename = "final_ratio_at_accuracy_" + str(format(curr_acc*100,".0f")) + ".pdf" 
        filename = prune_dir + "/" + filename
        plt.savefig(filename, dpi=600)
        plt.show()
        print("plot")
    
class Plots():
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
    
    def __PlotData(self,data, hue, title, run_type):
        plt.figure(figsize=(4.0, 2.8), dpi=600)
        #plt.subplot(1, 2, 1)
        plt.grid()
        #plt.rcParams.update({'font.family':'sans-serif'})
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams.update({'font.size': 8})
        ax = sns.lineplot(data=data, x="step", y="value", hue=hue, 
                          alpha=1, linewidth=0.8, ci=None)
        plt.legend(fontsize=6,loc='lower right')
        x_ticker = range(0,data.step.max(),10)
        ax.set_xticks(x_ticker)
        #y_ticker = np.around(np.linspace(0.7,1,30),decimals=2)
        start = math.floor(data.value.min()*100)
        y_ticker = [ x/100 for x in range(start,102,2)]
        ax.set_yticks(y_ticker)
        #ax.set_title(title)
        ylabel = ""
        if run_type == "train":
            ylabel = "Training Accuracy"
        elif run_type == "validation":
            ylabel = "Validation Accuracy"
        
        ax.set_ylabel(ylabel)
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
        
        epoch_interval = ["EpochInterval_2", "EpochInterval_10",
                         "EpochInterval_20"]
        total_pruning = ["TotalPruning_2", "TotalPruning_4","TotalPruning_6","TotalPruning_8",
                         "TotalPruning_10"]
        
        #self.__PlotResetNeuron("validation", prune_dir, 
        #                       epoch_interval[0], total_pruning[0])
        
        for ei in epoch_interval:
            for tp in total_pruning:
                self.__PlotResetNeuron("train",prune_dir, ei, tp)
                self.__PlotResetNeuron("validation",prune_dir, ei, tp)
        
        self.__PlotResetNeuron("train", prune_dir, "EpochInterval_40", "TotalPruning_3")
        self.__PlotResetNeuron("validation", prune_dir, "EpochInterval_40", "TotalPruning_3")
        
        reset_neuron = [True,False]
        
        for tp in total_pruning:
            for re in reset_neuron:
                self.__PlotNumberOfPruning("train",prune_dir, tp, re)
                self.__PlotNumberOfPruning("validation",prune_dir, tp, re)
        
        for ei in epoch_interval:
            for re in reset_neuron:
                self.__PlotEpochInterval("train", prune_dir, ei, re)
                self.__PlotEpochInterval("validation", prune_dir, ei, re)
        
    
    def __PlotResetNeuron(self, run_type, prune_dir, interval_type, prune_type):
        data = self.__GetData(run_type)
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
        
        self.__PlotData(data,hue_name, title, run_type)
        
        filename = ""
        if run_type == "train":
            filename =  prune_dir + "/TrainingAccuracy_ResetNeuron_" \
                        + interval_type + "_" + prune_type 
        elif run_type == "validation":
            filename =  prune_dir + "/ValidationAccuracy_ResetNeuron_" \
                        + interval_type + "_" + prune_type 
            
        self.__SavePlot(filename + ".pdf")

    def __PlotEpochInterval(self, run_type, prune_dir, interval_type, reset_neuron=False):
        data = self.__GetData(run_type)
        data = self.__FilterData(data,"tag","accuracy")
        standard_data = self.__FilterData(data,"run", "standard")
        std_hue_name = standard_data.run.apply(lambda label: "standard")
        if reset_neuron == True:
            data = self.__FilterData(data,"run","ResetNeuron")
        else:
            data = self.__ExcludeData(data,"run","ResetNeuron")
        optimal_data = self.__FilterData(data,"run","optimal")
        optimal_data = self.__FilterData(data,"run",interval_type)
        split_name = optimal_data.run.apply(lambda label: label.split("/")[0])
        split_name = split_name.apply(lambda label: label.split("_"))
        hue_name = split_name.apply(lambda label: "No. of Pruning:" + label[7])
        hue_name = hue_name.append(std_hue_name)
        data = optimal_data.append(standard_data)
        label = interval_type.split("_")
        title = "Epoch Interval:" + label[1] + ", Reset Neuron: "
        
        filename = ""
        if run_type == "train":
            filename =  prune_dir + "/TrainingAccuracy_" + interval_type  
        elif run_type == "validation":
            filename =  prune_dir + "/ValidationAccuracy_" + interval_type 
                        
        if reset_neuron == True:
            title += "Enabled"
            filename += "_ResetNeuron_Enabled"
        else:
            title += "Disabled"
            filename += "_ResetNeuron_Disabled"
        self.__PlotData(data,hue_name, title, run_type)
        self.__SavePlot(filename + ".pdf")

    def __PlotNumberOfPruning(self, run_type, prune_dir, 
                              total_pruning, reset_neuron=False):
        data = self.__GetData(run_type)
        data = self.__FilterData(data,"tag","accuracy")
        standard_data = self.__FilterData(data,"run", "standard")
        std_hue_name = standard_data.run.apply(lambda label: "standard")
        if reset_neuron == True:
            data = self.__FilterData(data,"run","ResetNeuron")
        else:
            data = self.__ExcludeData(data,"run","ResetNeuron")
        optimal_data = self.__FilterData(data,"run","optimal")
        optimal_data = self.__FilterData(data,"run",total_pruning)
        split_name = optimal_data.run.apply(lambda label: label.split("/")[0])
        split_name = split_name.apply(lambda label: label.split("_"))
        hue_name = split_name.apply(lambda label: "Epoch Interval:" + label[5])
        hue_name = hue_name.append(std_hue_name)
        data = optimal_data.append(standard_data)
        label = total_pruning.split("_")
        
        title = "No. of Pruning:" + label[1] + ", Reset Neuron: "
        
        filename = ""
        if run_type == "train":
            filename = prune_dir + "/TrainingAccuracy_" + total_pruning
        elif run_type == "validation":
            filename = prune_dir + "/ValidationAccuracy_" + total_pruning
            
        if reset_neuron == True:
            title += "Enabled"
            filename += "_ResetNeuron_Enabled"
        else:
            title += "Disabled"
            filename += "_ResetNeuron_Disabled"
        self.__PlotData(data,hue_name, title, run_type)
        self.__SavePlot(filename + ".pdf")
        
    def AnalyzeLoss(self, prune_dir):
        self.__CalculateEILoss("train", prune_dir)
        self.__CalculateEILoss("validation", prune_dir)
        self.__CalculateNumPruningLoss("train", prune_dir)
        self.__CalculateNumPruningLoss("validation", prune_dir)
        self.__CalculateResetNeuronLoss("train", prune_dir)
        self.__CalculateResetNeuronLoss("validation", prune_dir)
        
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
        self.__GeneratePlots(min_val_loss)
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
        self.__GeneratePlots(min_val_loss)
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
        #p_std_vs_rn = stats.ttest_ind(std_loss["value"], rn_loss["value"]) 
        run = pd.concat([std_run,run])
        min_val_loss['x'] = run
        self.__GeneratePlots(min_val_loss)
        self.__SavePlot(prune_dir + "/" + run_type + "_loss_ResetNeuron.pdf")
        
    def __GeneratePlots(self,min_val_loss):
        bp = sns.boxplot(data=min_val_loss, y="value", x=min_val_loss.x,
                         linewidth=1)
        #medians = min_val_loss.groupby(['x'])['value'].median()
        # offset from median for display
        #vertical_offset = min_val_loss['value'].median() * 0.002 
                
        #iterate over boxes
        for i,box in enumerate(bp.artists):
            box.set_edgecolor('black')
            box.set_facecolor('white')
            # iterate over whiskers and median lines
            for j in range(6*i,6*(i+1)):
                bp.lines[j].set_color('black')
        
        #medians = pd.to_numeric(medians.map('{:,.4f}'.format))
        #for xtick in bp.get_xticks():
        #    bp.text(xtick,medians[xtick] + vertical_offset,medians[xtick], 
        #    horizontalalignment='center',size='x-small',color='black',weight='semibold')
            
        bp.set(xlabel='Run',ylabel='Epoch Loss')
        
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