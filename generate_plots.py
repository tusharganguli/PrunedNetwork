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
import os
import utils

def generate_matrix_heatmap(dir_lst,title_lst, dir_name, prefix):    
    model_lst = []
    for d in dir_lst:
        model_lst.append(keras.models.load_model(d))
    
    plt.rc('xtick', labelsize=6)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=6)
    
    idx = 0
    num_layers = len(model_lst[idx].layers)
    for idx in range(num_layers):
        layer_lst = []
        for m in model_lst:
            layer_lst.append(m.layers[idx])
        
        if not isinstance(layer_lst[0],keras.layers.Dense):
            continue
        layer_name = layer_lst[0].name
        bool_lst = []
        for l in layer_lst:
            wts = l.get_weights()[0]
            dim = wts.shape
            s_bool = wts > 0
            bool_lst.append(np.invert(s_bool))
        
        subplot_cnt = len(dir_lst)
        fig, axs = plt.subplots(1,subplot_cnt)
        title = layer_name + ": " + str(dim[0]) + "x" + str(dim[1])
        fig.suptitle( title, fontsize=10)
        
        if subplot_cnt == 1:
            axs.set_title(title_lst[0])
            im = axs.imshow(bool_lst[0], cmap='hot', interpolation='nearest')
            im.set_clim(0,1)
        else:
            for s in range(subplot_cnt):
                axs[s].set_title(title_lst[s])
                im = axs[s].imshow(bool_lst[s], cmap='hot', interpolation='nearest')
                im.set_clim(0,1)
        
        filename = dir_name + "/" + prefix + "_" + layer_name + ".eps"
        plt.savefig(filename, dpi=600, format='eps')
        plt.show()
        

def matrix_heatmap(freq_model_dir, tf_model_dir, dir_name, prefix ):
    
    #std_model = keras.models.load_model(std_dir)    
    freq_model = keras.models.load_model(freq_model_dir)    
    tf_model = keras.models.load_model(tf_model_dir)
    
    plt.rc('xtick', labelsize=6)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=6)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams.update({'font.size': 10})
    
    idx = 0
    num_layers = len(freq_model.layers)
    for idx in range(num_layers):
        #s_layer = std_model.layers[idx]
        f_layer = freq_model.layers[idx]
        tf_layer = tf_model.layers[idx]
        
        if not isinstance(f_layer,keras.layers.Dense):
            continue
        layer_name = f_layer.name
        
        #s_wts = s_layer.get_weights()[0]
        #s_wts = np.abs(s_wts)
        #dim = s_wts.shape
        
        #s_bool = s_wts > 0
        #s_bool = np.invert(s_bool)
        
        f_wts = f_layer.get_weights()[0]
        f_wts = np.abs(f_wts)
        #dim = f_wts.shape
        
        f_bool = f_wts > 0
        f_bool = np.invert(f_bool)
        
        tf_wts = tf_layer.get_weights()[0]
        tf_wts = np.abs(tf_wts)
        tf_bool = tf_wts > 0
        tf_bool = np.invert(tf_bool)
        
        #c_bool = s_bool == f_bool
        
        fig, (ax1,ax2) = plt.subplots(1,2)
        #title = layer_name + ": " + str(dim[0]) + "x" + str(dim[1])
        #fig.suptitle( title, fontsize=10)
        
        #ax1.set_title('Standard')
        #im1 = ax1.imshow(s_bool, cmap='hot', interpolation='nearest')
        #im1.set_clim(0,1)
        
        ax1.set_title('Activation-based')
        im1 = ax1.imshow(f_bool, cmap='hot', interpolation='nearest')
        im1.set_clim(0,1)
        
        ax2.set_title('Magnitude-based')
        im2 = ax2.imshow(tf_bool, cmap='hot', interpolation='nearest')
        im2.set_clim(0,1)
        
        #axs[1,1].set_title('Difference')
        #im4 = axs[1,1].imshow(c_bool, cmap='hot', interpolation='nearest')
        
        filename = dir_name + "/" + prefix + "_" + layer_name + ".eps"
        plt.savefig(filename, dpi=600,format='eps')
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
        filename = "relative_ratio_at_accuracy_" + str(format(curr_acc*100,".0f")) + ".eps" 
        filename = prune_dir + "/" + filename
        plt.savefig(filename, dpi=600, format='eps')
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
        filename = "final_ratio_at_accuracy_" + str(format(curr_acc*100,".0f")) + ".eps" 
        filename = prune_dir + "/" + filename
        plt.savefig(filename, dpi=600, format='eps')
        plt.show()
        print("plot")
    
class Plots():
    
    def __init__(self, tbdev_file_name, prune_filename):
        
        self.df = pd.read_excel(tbdev_file_name)
        # the first row that we read from the excel file contains na values
        self.df=self.df.dropna(how='all')
        #self.pruning_df = pd.read_excel(prune_filename, usecols="A:E,G")
        self.pruning_df = pd.read_excel(prune_filename)
        self.pruning_df = self.pruning_df.dropna(how='all')
    
    def ConvertToEps(prune_dir):
        import glob, os
        os.chdir("./" + prune_dir)
        for file in glob.glob("*.pdf"):
            command = "inkscape " + file + " -o " + file.split(".")[0] + ".eps"
            os.system(command)
    
    def download_and_save(experiment_id, tbdev_dir, file_name):
        experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
        df = experiment.get_scalars()
        file_name = tbdev_dir + "/" + file_name + ".xls"
        utils.write(df,file_name)
        
        
        
    def __GetData(self, filter_value):
        data = self.df[self.df.run.str.contains(filter_value)]
        return data
    
    def __FilterData(self,data, col_name,filter_value):
        new_data = data[data[col_name].str.contains(filter_value)]
        return new_data
    
    def __ExcludeData(self,data, col_name,filter_value):
        new_data = data[~data[col_name].str.contains(filter_value)]
        return new_data
    
    def __PlotData(self,data, hue, title, run_type, plot_type, legend_title):
        plt.figure(figsize=(5.0, 4.0), dpi=600)
        #plt.subplot(1, 2, 1)
        plt.grid()
        #plt.rcParams.update({'font.family':'sans-serif'})
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams.update({'font.size': 10})
        
        ax = sns.lineplot(data=data, x="step", y="value", hue=hue, 
                          alpha=1, linewidth=0.8, ci=None)
        
        if plot_type == "loss":
            plt.legend(fontsize=6,loc='upper right', title=legend_title)
        else:
            plt.legend(fontsize=6,loc='lower right', title=legend_title)
        #elif plot_type == "loss":
            #plt.legend(fontsize=6,loc='upper right', title=legend_title)
            
        x_ticker = range(0,math.ceil(data.step.max()),10)
        ax.set_xticks(x_ticker)
        #y_ticker = np.around(np.linspace(0.7,1,30),decimals=2)
        step = 4
        #start = math.floor(data.value.min()*100)
        #end = math.ceil(data.value.max()*100) + step
        
        if plot_type == "accuracy":
            start = 50
            end = 100
        elif plot_type == "loss":
            step = 10
            start = 0
            end = 160
        else:
            raise ValueError("Illegal plot type passed.")
            
        y_ticker = [ x/100 for x in range(start,end,step)]
        ax.set_yticks(y_ticker)
        #ax.set_title(title)
        ylabel = ""
        if run_type == "train":
            ylabel = "Training"
        elif run_type == "validation":
            ylabel = "Validation"
        
        if plot_type == "accuracy":
            ylabel += " Accuracy"
        elif plot_type == "loss":
            ylabel += " Loss"
        
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Epoch")
        
    def __SavePlot(self, filename):
        
        dir_name = filename.rpartition("/")[0]
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        plt.savefig(filename, dpi=600, format='eps')
        plt.close()
        
    def __get_file_name(self, prune_dir, run_type, plot_type, filters):
        file_name = prune_dir + "/" + run_type + "_" + plot_type + "_"
        
        for f in filters:
            file_name +=  f + "_"
        file_name += ".eps"
        return file_name
                 
    def __get_legend_title(self, filter_lst, flags):
        legend_title = ""
        
        check = any(item in filter_lst for item in flags["neuron_update"])
        if not check:
            legend_title += "Neuron Update,"
        check = any(item in filter_lst for item in flags["prune_type"])
        if not check:
            legend_title += "Prune Type,Actual,"
        check = any(item in filter_lst for item in flags["prune_pct"])
        if not check:
            legend_title += "Total,Actual,"
        check = any(item in filter_lst for item in flags["num_pruning"])
        if not check:
            legend_title += "Pruning,"
        
        return legend_title[0:-1]
    
    def __get_true_pruning(self, row):
        pruning_df = self.pruning_df
        #if "ctr" in row:
        #    pruning_df = pruning_df[pruning_df["NeuronUpdate"] == "ctr"]
        #elif "activacc" in row: 
        #    pruning_df = pruning_df[pruning_df["NeuronUpdate"] == "activacc"]
        #elif "act" in row:
        #    pruning_df = pruning_df[pruning_df["NeuronUpdate"] == "act"]
        
        if "neurwts" in row:
            pruning_df = pruning_df[pruning_df["PruningType"] == "neurwts"]
        elif "neuron" in row:
            pruning_df = pruning_df[pruning_df["PruningType"] == "neuron"]
            
        if "PrunePct_80" in row:
            pruning_df = pruning_df[pruning_df["PrunePct"] == 80]
        elif "PrunePct_85" in row:
            pruning_df = pruning_df[pruning_df["PrunePct"] == 85]
        elif "PrunePct_90" in row:
            pruning_df = pruning_df[pruning_df["PrunePct"] == 90]
            
        if "NumPruning_1_" in row:
            pruning_df = pruning_df[pruning_df["NumPruning"] == 1]
        elif "NumPruning_5_" in row:
            pruning_df = pruning_df[pruning_df["NumPruning"] == 5]
        elif "NumPruning_10_" in row:
            pruning_df = pruning_df[pruning_df["NumPruning"] == 10]
        
        #if "ResetNeuron" in row:
        #    pruning_df = pruning_df[pruning_df["ResetNeuron"] == "Yes"]
        #else:
        #    pruning_df = pruning_df[pruning_df["ResetNeuron"] == "No"]
            
        return pruning_df["Pct Pruning"]
            
    def get_hue_name(self, row, filter_lst, flags):
        name = ""
        
        flag_lst = flags["neuron_update"]
        check = any(item in  flag_lst for item in filter_lst)
        if not check:
            if "ctr_" in flag_lst and "ctr_" in row:
                name += "Counter,"
            elif "activacc_" in flag_lst and "activacc_" in row:
                name += "Activation Accuracy,"
            elif "act_" in flag_lst and "act_" in row:
                name += "Activation,"
        
        flag_lst = flags["prune_type"]
        check = any(item in flag_lst for item in filter_lst)
        if not check:
            if "neurwts" in flag_lst and "neurwts" in row:
                name += "Neuron Wts,"
            elif "neuron" in flag_lst and "neuron" in row:
                name += "Neuron,"
            actual_pruning = self.__get_true_pruning(row)
            name += str(round(actual_pruning.to_numpy()[0],2)) + "%,"
        
        flag_lst = flags["prune_pct"]
        check = any(item in  flag_lst for item in filter_lst)
        if not check:
            if "PrunePct_80" in flag_lst and "PrunePct_80" in row:
                name += "80%," 
            elif "PrunePct_85" in flag_lst and "PrunePct_85" in row:
                name += "85%,"
            elif "PrunePct_90" in flag_lst and "PrunePct_90" in row:
                name += "90%,"
            actual_pruning = self.__get_true_pruning(row)
            name += str(round(actual_pruning.to_numpy()[0],2)) + "%,"
            
        flag_lst = flags["num_pruning"]
        check = any(item in  flag_lst for item in filter_lst)
        if not check:
            if "NumPruning_1_" in flag_lst and "NumPruning_1_" in row:
                name += "1,"
            elif "NumPruning_5_" in flag_lst and "NumPruning_5_" in row:
                name += "5,"
            elif "NumPruning_10_" in flag_lst and "NumPruning_10_" in row:
                name += "10,"
            
            
        return name[0:-1]
    
    def __get_hue_names(self, data, filter_lst, flags):
        
        run_name = data.run.apply(lambda label: label.split("/")[0])
        hue_names = [self.get_hue_name(r,filter_lst,flags) for r in run_name]
        legend_title = self.__get_legend_title(filter_lst,flags)
        #hue_name = hue_name.apply(lambda label: name if n_update in label && p_type in label and p_pct in label and n_pruing in label)
        return (hue_names,legend_title)
    
    def __Plot(self, run_type, plot_type, title, filter_lst, exclude_lst, flags):
        data = self.__GetData(run_type)
        data = self.__FilterData(data,"tag", plot_type)
        filter_data = data
        for fil in filter_lst:
            filter_data = self.__FilterData(filter_data,"run", fil)
        
        for exc in exclude_lst:
            filter_data = self.__ExcludeData(filter_data,"run", exc)
            
        hue_names, legend_title = self.__get_hue_names(filter_data, filter_lst, 
                                                       flags)
        
        self.__PlotData(filter_data, hue_names, title, 
                        run_type, plot_type, legend_title)

    def PlotIntervalPruning(self, prune_dir):
        # , "activacc_","act_"
        neuron_update = ["ctr_"]
        prune_type = ["neurwts","neuron"]
        # ,"PrunePct_90"
        prune_pct = ["PrunePct_80","PrunePct_85"]
        # "NumPruning_1_"
        num_pruning = ["NumPruning_10_","NumPruning_5_"]
        
        flags = {}
        flags["neuron_update"] = neuron_update
        flags["prune_type"] =prune_type
        flags["prune_pct"] = prune_pct
        flags["num_pruning"] = num_pruning
        
        exclude_lst = ["ResetNeuron"]
        exclude_lst.append("NumPruning_1_")
        #"""
        for p_pct in prune_pct:
            #for n_pruning in num_pruning:
            for n_update in neuron_update:
                title = "Neuron Update"
                #filter_lst = [p_pct,n_pruning,n_update]
                filter_lst = [p_pct,n_update]
                
                run_type = "train"
                plot_type = "accuracy"
                self.__Plot(run_type, plot_type, title, 
                            filter_lst, exclude_lst, flags)
                file_name = self.__get_file_name(prune_dir, run_type, 
                                                 plot_type, filter_lst)
                self.__SavePlot(file_name)
                
                plot_type = "loss"
                self.__Plot(run_type, plot_type, title, 
                            filter_lst, exclude_lst, flags)
                file_name = self.__get_file_name(prune_dir, run_type, 
                                                 plot_type, filter_lst)
                self.__SavePlot(file_name)
                
                run_type = "validation"
                plot_type = "accuracy"
                self.__Plot(run_type, plot_type, title, 
                            filter_lst, exclude_lst, flags)
                file_name = self.__get_file_name(prune_dir, run_type, 
                                                 plot_type, filter_lst)
                self.__SavePlot(file_name)
                
                plot_type = "loss"
                self.__Plot(run_type, plot_type, title, 
                            filter_lst, exclude_lst, flags)
                file_name = self.__get_file_name(prune_dir, run_type, 
                                                 plot_type, filter_lst)
                self.__SavePlot(file_name)
            
        #"""     
        #exclude_lst.append("NumPruning_1_")
        exclude_lst.append("PrunePct_90")
        
        for n_update in neuron_update:
            for p_type in prune_type:
                title = "Neuron Update"
                filter_lst = [n_update, p_type]
                
                run_type = "train"
                plot_type = "accuracy"
                self.__Plot(run_type, plot_type, title, 
                            filter_lst, exclude_lst, flags)
                file_name = self.__get_file_name(prune_dir, run_type, 
                                                 plot_type, filter_lst)
                self.__SavePlot(file_name)
                
                #plot_type = "loss"
                #self.__Plot(run_type, plot_type, title, 
                #            filter_lst, exclude_lst, flags)
                #file_name = self.__get_file_name(prune_dir, run_type, 
                #                                 plot_type, filter_lst)
                #self.__SavePlot(file_name)
        
                run_type = "validation"
                plot_type = "accuracy"
                self.__Plot(run_type, plot_type, title, 
                            filter_lst, exclude_lst, flags)
                file_name = self.__get_file_name(prune_dir, run_type, 
                                                 plot_type, filter_lst)
                self.__SavePlot(file_name)
                
                #plot_type = "loss"
                #self.__Plot(run_type, plot_type, title, 
                #            filter_lst, exclude_lst, flags)
                #file_name = self.__get_file_name(prune_dir, run_type, 
                #                                 plot_type, filter_lst)
                #self.__SavePlot(file_name)
             
        
        #"""
        run_type = "train"
        self.__CalculateResetNeuronLoss(run_type, prune_dir)
        run_type = "validation"
        self.__CalculateResetNeuronLoss(run_type, prune_dir)
        #"""
                   
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
            
        self.__SavePlot(filename + ".eps")

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
        self.__SavePlot(filename + ".eps")

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
        self.__SavePlot(filename + ".eps")
        
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
        self.__SavePlot(prune_dir + "/" + run_type + "_loss_EpochInterval.eps")
        
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
        self.__SavePlot(prune_dir + "/" + run_type + "_loss_NumPruning.eps")
        
    def __CalculateResetNeuronLoss(self, run_type, prune_dir):
        data = self.__GetData(run_type)
        data = self.__FilterData(data,"tag","loss")
        mean_loss = data.groupby("run", as_index=False).agg({"value": "mean"})
        min_loss = data.groupby("run", as_index=False).agg({"value": "min"})
        max_loss = data.groupby("run", as_index=False).agg({"value": "max"})
        
        #opt_data = self.__FilterData(min_loss,"run","optimal_")
        mean_rn_loss = self.__FilterData(mean_loss,"run","ResetNeuron_")
        mean_no_rn_loss = self.__ExcludeData(mean_loss,"run","ResetNeuron_")
        min_rn_loss = self.__FilterData(min_loss,"run","ResetNeuron_")
        min_no_rn_loss = self.__ExcludeData(min_loss,"run","ResetNeuron_")
        max_rn_loss = self.__FilterData(max_loss,"run","ResetNeuron_")
        max_no_rn_loss = self.__ExcludeData(max_loss,"run","ResetNeuron_")
        
        mean_val_loss = pd.concat([mean_rn_loss, mean_no_rn_loss])
        min_val_loss = pd.concat([min_rn_loss, min_no_rn_loss])
        max_val_loss = pd.concat([max_rn_loss, max_no_rn_loss])
        
        mean_run = mean_val_loss.run.apply(lambda label: label.split("/")[0])
        min_run = min_val_loss.run.apply(lambda label: label.split("/")[0])
        max_run = max_val_loss.run.apply(lambda label: label.split("/")[0])
        
        mean_run = mean_run.apply(lambda label: "Reset Enabled" 
                        if "ResetNeuron" in label  else "Reset Disabled" )
        min_run = min_run.apply(lambda label: "Reset Enabled" 
                        if "ResetNeuron" in label  else "Reset Disabled" )
        max_run = max_run.apply(lambda label: "Reset Enabled" 
                        if "ResetNeuron" in label  else "Reset Disabled" )
        
        #std_loss = self.__FilterData(min_loss,"run","standard_")
        #std_run = std_loss.run.apply(lambda label: "Standard")
        #min_val_loss = pd.concat([std_loss, min_val_loss])
        #p_std_vs_rn = stats.ttest_ind(std_loss["value"], rn_loss["value"]) 
        #run = pd.concat([std_run,run])
        mean_val_loss['x'] = mean_run
        min_val_loss['x'] = min_run
        max_val_loss['x'] = max_run
        
        self.__GeneratePlots(mean_val_loss)
        self.__SavePlot(prune_dir + "/" + run_type + "_mean_loss_ResetNeuron.eps")
        self.__GeneratePlots(min_val_loss)
        self.__SavePlot(prune_dir + "/" + run_type + "_min_loss_ResetNeuron.eps")
        self.__GeneratePlots(max_val_loss)
        self.__SavePlot(prune_dir + "/" + run_type + "_max_loss_ResetNeuron.eps")
        
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
            
        bp.set(xlabel='Run',ylabel='Mean Epoch Loss')
        
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