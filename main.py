#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:54:36 2021

@author: tushar
"""
from tensorflow import keras
import tf_pruning as tfp
import glob
import os
import pandas as pd
import numpy as np

import model_run as mr
import generate_plots as gp
import utils

db = keras.datasets.fashion_mnist
f_acc = 0.98
model_dir = "model"
tensorboard_dir = "tb_results"
prune_dir = "prune_details"
prune_trained_model_dir = model_dir
train_pruned_model_dir = model_dir
plot_dir = "plots"

# generating the plots after uploading the plots in the 
# tensorboard dev board

"""
prune_dir = "optimal_prune_details"
experiment_id = "VToMCmUyQPOlxHA2KAdkAQ"
plots = gp.Plots(experiment_id)
#plots.PlotOptimal(prune_dir)
#plots.AnalyzeLoss(prune_dir)
plots.ConvertToEps(prune_dir)
"""

def generate_tb_data():
    prune_dir = "cip/tb_dev_results"
    experiment_id = "Qf3qKjq6Rxu9iG5LLUOnWQ"
    #plots = gp.Plots(experiment_id)
    #plots.PlotIntervalPruning(prune_dir)
    gp.Plots.ConvertToEps(prune_dir)
    
#generate_tb_data()



std_full_dense_dir = model_dir + "/standard_full_dense"
std_dir = model_dir + "/standard"
run_cnt = 3

def train_standard():
    log_dir = "standard"
    model_run = mr.ModelRun(db,log_dir, tensorboard_dir, 
                            prune_dir, model_dir,
                            plot_dir)
    
    model_run.set_log_filename("standard")
    
    #neuron_lst = [300,300,300]
    #model = model_run.create_full_dense_model("standard",neuron_lst)
    
    model_run.evaluate_standard(run_type="standard",
                                num_runs=run_cnt, 
                                final_acc = f_acc,
                                es_delta = 0.1)
    model_run.save_log()
    model_name = model_run.get_modelname()
    model_run.save_model(model_name)
    del model_run

#train_standard()

def cnn_pruning():
    """
    Evaluates pruning after reaching specific training accuracy 
    for each stage of pruning    

    Returns
    -------
    None.

    """
    log_dir = "cnn"
    model_run = mr.ModelRun(db)
    
    log_handler = utils.LogHandler(log_dir, tensorboard_dir, prune_dir, 
                               model_dir, plot_dir)
    
    prune_start_at = 80/100
    n_pruning = 3
    f_acc = 98/100
    prune_pct = 80
    #r_neuron = False
    delta = 0.1
    
    # neuron_update: ctr,act,act_acc
    #n_update = "act"
    # pruning_type: neuron, neuron_wts
    #p_type = "neuron"
    
    neuron_update = "ctr"
    prune_type = "neuron_wts"
    reset_neuron = False
    
    model_name = "cnn" 
    #model_name += "_PruneStart_" + str(prune_start_at)
    model_name += "_NumPruning_" + str(n_pruning)
    model_name += "_PrunePct_" + str(prune_pct)
    #model_name += "_FinalAcc_" + str(f_acc)
    model_name += "_NeuronUpdate_" + neuron_update
    model_name += "_PruningType_" + prune_type
    if reset_neuron == True:
        model_name += "_ResetNeuron"
    model_name += "_EarlyStopping_" + str(delta)
    
    model_name = utils.add_time_to_filepath(model_name)
    
    log_handler.set_log_filename(model_name)
    
    #model_run.set_log_handler(log_handler)
    
    model_run.evaluate_cnn(run_type="cnn", 
                        prune_start_at_acc = prune_start_at,
                        num_pruning = n_pruning,
                        final_acc = f_acc,
                        prune_pct = prune_pct,
                        neuron_update = neuron_update,
                        pruning_type = prune_type,
                        reset_neuron = reset_neuron,
                        early_stopping_delta=delta,
                        log_handler = log_handler)
    
    #log_handler.log_single_run(model_name)
    
    prune_model_name = log_handler.get_modelname()
    model_run.save_model(prune_model_name)
    
    prune_filename = utils.add_time_to_filepath("prune_details")
    log_handler.set_log_filename(prune_filename)
    log_handler.write_to_file(prune_filename)
    del model_run

#cnn_pruning()

def cip_pruning():
    """
    Evaluates pruning after reaching specific training accuracy 
    for each stage of pruning    

    Returns
    -------
    None.

    """
    log_dir = "cip"
    model_run = mr.ModelRun(db)
    
    log_handler = utils.LogHandler(log_dir, tensorboard_dir, prune_dir, 
                               model_dir, plot_dir)
    
    prune_start_at = 80/100
    n_pruning_lst = [10]
    f_acc = 98/100
    prune_pct_lst = [80,85,90]
    #r_neuron = False
    delta = 0.1
    
    # neuron_update: ctr,act,act_acc
    #n_update = "act"
    # pruning_type: neuron, neuron_wts
    #p_type = "neuron"
    
    neuron_update_lst = ["ctr","act","act_acc"]
    prune_type_lst = ["neuron","neuron_wts"]
    reset_neuron_lst = [False]
    
    for r_neuron in reset_neuron_lst:
        for n_pruning in n_pruning_lst:
            for n_update in neuron_update_lst:
                for p_type in prune_type_lst:
                    for p_pct in prune_pct_lst:
                        model_name = "cip" 
                        #model_name += "_PruneStart_" + str(prune_start_at)
                        model_name += "_NumPruning_" + str(n_pruning)
                        model_name += "_PrunePct_" + str(p_pct)
                        #model_name += "_FinalAcc_" + str(f_acc)
                        model_name += "_NeuronUpdate_" + n_update
                        model_name += "_PruningType_" + p_type
                        if r_neuron == True:
                            model_name += "_ResetNeuron"
                        model_name += "_EarlyStopping_" + str(delta)
                        
                        model_name = utils.add_time_to_filepath(model_name)
                        
                        log_handler.set_log_filename(model_name)
                        
                        #model_run.set_log_handler(log_handler)
                        
                        model_run.evaluate_cip(run_type="cip", 
                                               prune_start_at_acc = prune_start_at,
                                               num_pruning = n_pruning,
                                               final_acc = f_acc,
                                               prune_pct = p_pct,
                                               neuron_update = n_update,
                                               pruning_type = p_type,
                                               reset_neuron = r_neuron,
                                               early_stopping_delta=delta,
                                               log_handler = log_handler)
                        
                        log_handler.log_single_run(model_name)
                        
                        prune_model_name = log_handler.get_modelname()
                        model_run.save_model(prune_model_name)
                        
    prune_filename = utils.add_time_to_filepath("prune_details")
    log_handler.set_log_filename(prune_filename)
    log_handler.write_to_file(prune_filename)
    del model_run

#cip_pruning()

def cip_test():
    log_dir = "cip"
    model_run = mr.ModelRun(db)
    log_handler = utils.LogHandler(log_dir, tensorboard_dir, prune_dir, 
                               model_dir, plot_dir)
    
    
    prune_start_at = 80/100
    n_pruning = 1
    f_acc = 98/100
    prune_pct = 80
    r_neuron = False
    delta = 0.1
    
    # neuron_update: ctr,act,act_acc
    n_update = "act"
    # pruning_type: neuron, neuron_wts
    p_type = "neuron"
    
    model_name = "cip" 
    #model_name += "_PruneStart_" + str(prune_start_at)
    model_name += "_NumPruning_" + str(n_pruning)
    model_name += "_PrunePct_" + str(prune_pct)
    #model_name += "_FinalAcc_" + str(f_acc)
    model_name += "_NeuronUpdate_" + n_update
    model_name += "_PruningType_" + p_type
    if r_neuron == True:
        model_name += "_ResetNeuron"
    model_name += "_EarlyStopping_" + str(delta)
    
    model_name = utils.add_time_to_filepath(model_name)
    
    log_handler.set_log_filename(model_name)
    
    model_run.evaluate_cip(run_type="cip", 
                           prune_start_at_acc = prune_start_at,
                           num_pruning = n_pruning,
                           final_acc = f_acc,
                           prune_pct = prune_pct,
                           neuron_update = n_update,
                           pruning_type = p_type,
                           reset_neuron = r_neuron,
                           early_stopping_delta=delta,
                           log_handler = log_handler)
    
    prune_filename = utils.add_time_to_filepath("prune_details")
    log_handler.set_log_filename(prune_filename)
    log_handler.write_to_file(prune_filename)
    del model_run

#cip_test()

def prune_standard_tf(trained_model_dir=std_dir,
                      tf_pruned_model_dir = model_dir + "/tf_pruned"):
    trained_model = keras.models.load_model(trained_model_dir)    
    
    initial_sparsity=0
    final_sparsity=0.80
    tensorboard_dir = "tf_prune_results"
    tf_prune = tfp.TFPrune(db, tensorboard_dir)
    (history, test_result, tf_pruned_model) = tf_prune.prune_dense(trained_model, initial_sparsity, final_sparsity)
    
    tf_pruned_model.save(tf_pruned_model_dir)
    filename = model_dir+ "/tf_prune_history.xlsx"
    tf_prune.log(history, test_result, filename)

tf_pruned_model_dir = model_dir + "/tf_pruned"
#prune_standard_tf()


def prune_trained(trained_model_dir = model_dir + "/standard"):    
    
    log_dir = "prune_trained"
    
    r_type = "prune_trained"
    n_pruning = 1
    p_int = 1
    f_acc = 93/100
    p_pct = 80
    r_neuron = False
    
    # neuron_update: ctr,act,act_acc
    #n_update = "act_acc"
    # pruning_type: neuron, neuron_wts
    #p_type = "neuron_wts"
    
    neuron_update_list = ["ctr","act","act_acc"]
    prune_type_list = ["neuron","neuron_wts"]
    
    model_run = mr.ModelRun(db,log_dir, tensorboard_dir, 
                            prune_dir, model_dir,
                            plot_dir)
    
    
    for n_update in neuron_update_list:
        for p_type in prune_type_list:
            model_name = r_type
            model_name += "_NumPruning_" + str(n_pruning)
            model_name += "_PruneInterval_" + str(p_int)
            model_name += "_FinalAcc_" + str(f_acc)
            model_name += "_NeuronUpdate_" + n_update
            model_name += "_PruningType_" + p_type
            if r_neuron == True:
                model_name += "_ResetNeuron"
            
            model_run.set_log_filename(model_name)
            
            trained_model = keras.models.load_model(trained_model_dir)
            model_run.prune_trained_model(run_type=r_type,
                                          trained_model=trained_model,
                                          num_pruning=n_pruning,
                                          pruning_interval=p_int,
                                          final_acc=f_acc,
                                          prune_pct=p_pct,
                                          neuron_update = n_update,
                                          prune_type = p_type,
                                          reset_neuron = r_neuron)
            
            prune_model_name = model_run.get_modelname()
            model_run.save_model(prune_model_name)
            del trained_model

trained_model_dir = model_dir + "/standard"    
#prune_trained( trained_model_dir )


def train_pruned(tf_model_dir):
    log_dir = "train_pruned"
    r_type="train_pruned"
    final_acc = 0.98
    filename = "tf_pruned"
    
    model_run = mr.ModelRun(db,log_dir, tensorboard_dir, prune_dir, model_dir,
                            plot_dir)
    
    tf_pruned_model = keras.models.load_model(tf_model_dir) 
    
    model_run.train_pruned_model(r_type,tf_pruned_model, final_acc, filename)
    model_name = model_run.get_modelname()
    model_run.save_model(model_name)
    
    for m_dir in glob.glob("./prune_trained/model/*"):    
        freq_pruned_model = keras.models.load_model(m_dir)
        filename = m_dir.split("/")[-1]
        model_run.train_pruned_model(r_type,freq_pruned_model, final_acc, filename)
        model_name = model_run.get_modelname()
        model_run.save_model(model_name)
    
#freq_pruned_dir = model_dir + "/fwp_prune_dense_test"
tf_pruned_dir = model_dir + "/tf_pruned"
#freq_trained_dir = model_dir + "/trained_fwp_prune_dense_test"
#tf_trained_dir = model_dir + "/train_tf_dense_pruned"

#train_pruned(tf_pruned_dir)


def matrix_norms(std_dir, tf_dir):
    std_model = keras.models.load_model(std_dir)
    tf_model = keras.models.load_model(tf_dir)
    matrix_norm_dir = "./train_pruned/matrix_norm/"
    if not os.path.exists(matrix_norm_dir):
        os.makedirs(matrix_norm_dir)

    for trained_model_dir in glob.glob("./train_pruned/model/*"):
        norm_file = matrix_norm_dir + trained_model_dir.split("/")[-1] + ".xlsx"        
        trained_model = keras.models.load_model(trained_model_dir)    
        df = utils.generate_matrix_norms(std_model, trained_model, tf_model)    
        df.to_excel(norm_file)      


std_dir = model_dir + "/standard"
#freq_dir = model_dir + "/cip_NumPruning_2_PruningInterval_1_FinalAcc_0.92_TotalPrune_80"
tf_dir = model_dir + "/tf_pruned"
#norm_file = model_dir+"/matrix_norm_standard_pruned.xlsx"

#matrix_norms(std_dir, tf_dir)


def matrix_heatmap():    
    matrix_heatmap_dir = "./train_pruned/matrix_heatmap/"
    if not os.path.exists(matrix_heatmap_dir):
        os.makedirs(matrix_heatmap_dir)
    
    svd_plots = gp.SVDPlots()
    
    for trained_model_dir in glob.glob("./train_pruned/model/*"):
        dir_name = matrix_heatmap_dir + trained_model_dir.split("/")[-1]
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        gp.matrix_heatmap( std_dir, trained_model_dir,
                          tf_dir, dir_name)
        svd_plots.ConvertToEps(dir_name)

#matrix_heatmap()


def get_modelname(m_dir):
    model_name = m_dir.split("/")[-1]
    freq_name = ""
    if "_act_acc_" in model_name:
        freq_name = "Activation and Frequency"
    elif "_act_" in model_name:
        freq_name = "Activation"
    else:
        freq_name = "Counter"
        
    prune_name = ""
    if "_neuron_wts" in model_name:
        prune_name = "Neuron and Weights"
    else:
        prune_name = "Neuron"

    return (prune_name, freq_name)
    
def get_std_rank(std_loc):
    model = keras.models.load_model(std_loc)
    prune_name = ""
    freq_name = ""
    
    data = pd.DataFrame([[prune_name,freq_name]], 
                            columns=["Prune Type", "Neuron Update"]) 
    
    for layer in model.layers:
        if not isinstance(layer,keras.layers.Dense):
            continue
        wts = layer.weights
        rank = np.linalg.matrix_rank(wts[0])    
        data[layer.name] = rank
    return data
    
def model_ranks(model_loc, filename):
    
    df = pd.DataFrame(columns=["Prune Type", "Neuron Update"])
    
    std_df = get_std_rank("./model/standard")
    df = df.append(std_df)
    
    cwd = os.getcwd()
    
    os.chdir(model_loc)
    
    for m_dir in glob.glob("*"):
        model = keras.models.load_model(m_dir)
        prune_name, freq_name = get_modelname(m_dir)
        
        data = pd.DataFrame([[prune_name,freq_name]], 
                                columns=["Prune Type", "Neuron Update"]) 
        
        for layer in model.layers:
            if not isinstance(layer,keras.layers.Dense):
                continue
            wts = layer.weights
            rank = np.linalg.matrix_rank(wts[0])    
            data[layer.name] = rank
        
        df = df.append(data)
    os.chdir(cwd)
    utils.write(df,filename)

def model_rank(model_dir, filename):
    df = pd.DataFrame(columns=["Prune Type", "Neuron Update"])
    model_df = get_std_rank(model_dir)
    df = df.append(model_df)
    utils.write(df,filename)
    
root_dir = "./prune_trained"
model_loc = root_dir + "/model"
filename = root_dir + "/rank.xlsx"
#model_ranks(model_loc, filename)

model_name = "cip_NumPruning_5_NeuronUpdate_ctr_PruningType_neuron_wts_2022_07_30-12_32_45"
model_dir = "./cip/model/" + model_name
filename = model_dir + ".xlsx"
#model_rank(model_dir, filename)


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


def optimal_pruning():

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
import tempfile
tensorboard_dir = tempfile.mkdtemp()
prune_dir = tempfile.mkdtemp()
model = mr.ModelRun(db,tensorboard_dir, prune_dir)

result = model.evaluate_low_rank_approx(std_model, cip_model)  
"""

"""
Evaluates one time pruning
"""
"""
tensorboard_dir = "otp_results"
prune_dir = "otp_prune_details"
model = mr.ModelRun(db,tensorboard_dir, prune_dir)
one_time_neuron_update = 1
"""
"""
model.evaluate_otp(run_type="otp", 
                   num_runs=run_cnt, 
                   pruning_type="weights",
                   neuron_update_at_acc = 85/100,
                   target_prune_pct=90,
                   prune_at_accuracy=90/100,
                   final_training_accuracy = t_acc
                   )


"""
"""
# neuron update from beginning
for tp_pct in [70,80,90]:
    for p_at_acc in range(90,98,2):
        model.evaluate_otp(run_type="otp", 
                           num_runs=run_cnt, 
                           pruning_type="weights",
                           neuron_update_at_acc = 5/100,
                           target_prune_pct=tp_pct,
                           prune_at_accuracy=p_at_acc/100,
                           final_training_accuracy = t_acc
                           )

model.write_to_file(filename = "OTP_EarlyNeuronUpdatePruning90AndMore.xls")

# neuron update later
for tp_pct in [70,80,90]:
    for p_at_acc in range(90,98,2):
        model.evaluate_otp(run_type="otp", 
                           num_runs=run_cnt, 
                           pruning_type="weights",
                           neuron_update_at_acc = (p_at_acc-5)/100,
                           target_prune_pct=tp_pct,
                           prune_at_accuracy=p_at_acc/100,
                           final_training_accuracy = t_acc
                           )

model.write_to_file(filename = "OTP_LateNeuronUpdatePruning90AndMore.xls")
"""

"""
# pruning weights based on one time update of the neuron frequency
for tp_pct in [70,80,90]:
    for p_at_acc in range(95,15,-10):
        model.evaluate_otp(run_type="otp", 
                           num_runs=run_cnt, 
                           pruning_type="weights",
                           neuron_update_at_acc = one_time_neuron_update,
                           target_prune_pct=tp_pct,
                           prune_at_accuracy=p_at_acc/100,
                           final_training_accuracy = t_acc)
model.write_to_file(filename = "OTP_OneTimeNeuronUpdate.xls")
"""      

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

