#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:54:36 2021

@author: tushar
"""
import sys
from tensorflow import keras
if sys.version_info.major == 3 and sys.version_info.minor == 8:
    import tf_pruning as tfp
import glob
import os
import pandas as pd
import numpy as np

import model_run as mr
import generate_plots as gp
import utils
import data
import pca_analysis as pa



db = keras.datasets.fashion_mnist
f_acc = 0.98
root_log_dir = "../LogDir"
model_dir = "model"
tensorboard_dir = "tb_results"
prune_dir = "prune_details"
prune_trained_model_dir = model_dir
train_pruned_model_dir = model_dir
plot_dir = "plots"

def train_pruned(pruned_model_name):
    log_dir = "/train_pruned"
    r_type="train_pruned"
    final_acc = 0.98
    log_filename = pruned_model_name + "train_pruned"
    es_delta = 0.1
    
    model_run = mr.ModelRun(db)
    log_handler = utils.LogHandler(log_dir, tensorboard_dir, prune_dir, 
                               model_dir, plot_dir)
    
    model_name = utils.add_time_to_filename(log_filename)
    log_handler.set_log_filename(model_name)
    
    pruned_model = keras.models.load_model(pruned_model_name) 
    
    model_run.train_pruned_model(r_type,pruned_model, 
                                 final_acc, es_delta, log_handler)
    model_name = log_handler.get_modelname()
    model_run.save_model(model_name)
    
    del pruned_model
    
    prune_filename = utils.add_time_to_filename("prune_details")
    log_handler.set_log_filename(prune_filename)
    log_handler.write_to_file(prune_filename)
    
    return model_name

# root_log_dir   
#freq_pruned_model_dir = "./prune_trained/model/prune_trained_NumPruning_5_NeuronUpdate_ctr_PruningType_neuron_wts_EarlyStopping_0.1_2022_08_19-12_02_11"
#tf_pruned_model_dir = "./tf_prune_trained/model"
#train_pruned(freq_pruned_model_dir, tf_pruned_model_dir)

def __get_rank(model, model_name, processing, df_rank):
    rank_lst = []
    rank_lst.append(model_name)
    rank_lst.append(processing)
    
    for layer in model.layers:
        if not isinstance(layer,keras.layers.Dense) or layer.name == "output":
            continue
        wts = layer.weights
        rank = np.linalg.matrix_rank(wts[0])
        rank_lst.append(rank)
        
    # Append the list as a new row to the DataFrame
    #df_rank = df_rank.append(pd.Series(rank_lst, index=df_rank.columns), ignore_index=True)
    
    # Convert the list to a DataFrame
    new_row_df = pd.DataFrame([rank_lst], columns=df_rank.columns)

    # Concatenate the new row DataFrame with the original DataFrame
    df_rank = pd.concat([df_rank, new_row_df], ignore_index=True)
    return df_rank

def get_rank_pre_post_training():
    #prune_dir = root_log_dir + "/cip/test"
    prune_dir = root_log_dir + "/pre_post_training"
    
    cwd = os.getcwd()
    os.chdir(prune_dir)
    
    # Column names for the DataFrame
    column_names = ['Model Name','Processing','Layer 1', 'Layer 2', 'Layer 3', 'Layer 4']

    # Create an empty DataFrame with the specified column names
    df_rank = pd.DataFrame(columns=column_names)
    
    for m_dir in glob.glob("*"):
        model = keras.models.load_model(m_dir)
        df_rank = __get_rank(model, m_dir, "Pre", df_rank)
        model_name = train_pruned(m_dir)
        model = keras.models.load_model(model_name)
        df_rank = __get_rank(model, m_dir, "Post", df_rank)
        del model
        
    file_path = "../rank_pre_post_training.xlsx"
    # Write the DataFrame to Excel
    df_rank.to_excel(file_path, index=False)
    os.chdir(cwd)
get_rank_pre_post_training()

    
def train_lowrank_models(database):
    import model_run as mr
    
    cwd = os.getcwd()
    os.chdir(pa.lrm_dir)
    
    for m_dir in glob.glob("*"):
        if "standard" in m_dir:
            continue
        log_dir = "train_pca"
        r_type="train_pca"
        final_acc = 0.98
        es_delta = 0.1
        
        model_run = mr.ModelRun(database)
        log_handler = utils.LogHandler(log_dir, tensorboard_dir, prune_dir, 
                                   model_dir, plot_dir)
        
        model_name = utils.add_time_to_filename(m_dir)
        model_name  = "pca_" + model_name
        log_handler.set_log_filename(model_name)
        
        model = keras.models.load_model(m_dir) 
        
        model_run.train_pruned_model(r_type,model, final_acc, es_delta, log_handler)
        model_name = log_handler.get_modelname()
        model_run.save_model(model_name)
        
        del model
        
    os.chdir(cwd)
    
#train_lowrank_models(db)

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
    experiment_id = "Qf3qKjq6Rxu9iG5LLUOnWQ"
    tbdev_dir = "cip/tb_dev_data"
    file_name="IntervalPruning"
    #gp.Plots.download_and_save(experiment_id, tbdev_dir, file_name)
    
    
    tbdev_plot_dir = "cip/tb_dev_results"
    tbdev_file_name = tbdev_dir + "/" + file_name + ".xls"
    prune_filename = "cip/" + prune_dir + "/prunedetails.xlsx"
    plots = gp.Plots(tbdev_file_name, prune_filename)
    plots.PlotIntervalPruning(tbdev_plot_dir)
    gp.Plots.ConvertToEps(tbdev_plot_dir)
    
#generate_tb_data()


def layer_wise_pruning():
    """
    prunes a single layer each time    

    Returns
    -------
    None.

    """
    log_dir = "layer_prune"
    model_run = mr.ModelRun(db)
    
    log_handler = utils.LogHandler(log_dir, tensorboard_dir, prune_dir, 
                               model_dir, plot_dir)
    
    prune_start_at = 80/100
    f_acc = 98/100
    prune_pct_lst = [80,85,90]
    delta = 0.1
    prune_type = ["neuron","neuron_wts"]
    reset_neuron = False
    
    
    #model_run.set_log_handler(log_handler)
    
    for ptype in prune_type:
        for pct in prune_pct_lst:
            model_name = "layer_wise" 
            model_name += "_PruningType_" + ptype
            model_name += "_PrunePct_" + str(pct)
            model_name += "_EarlyStopping_" + str(delta)
            
            model_name = utils.add_time_to_filename(model_name)
            
            log_handler.set_log_filename(model_name)
            
            model_run.evaluate_layer_wise(run_type="layer_wise", 
                                          prune_start_at_acc = prune_start_at,
                                          final_acc = f_acc,
                                          prune_pct = pct,
                                          neuron_update = "ctr",
                                          pruning_type = ptype,
                                          reset_neuron = reset_neuron,
                                          early_stopping_delta=delta,
                                          log_handler = log_handler)
            
            log_handler.log_single_run(model_name)
            
            prune_model_name = log_handler.get_modelname()
            model_run.save_model(prune_model_name)
                        
    prune_filename = utils.add_time_to_filename("prune_details")
    log_handler.set_log_filename(prune_filename)
    log_handler.write_to_file(prune_filename)
    del model_run

#layer_wise_pruning()


#std_dir = model_dir + "/standard"
run_cnt = 3

def train_standard():
    log_dir = "standard"
    model_run = mr.ModelRun(db)
    
    log_handler = utils.LogHandler(log_dir, tensorboard_dir, prune_dir, 
                                   model_dir, plot_dir)
    
    model_name = "standard"
    model_name = utils.add_time_to_filename(model_name)
    log_handler.set_log_filename(model_name)
    
    #neuron_lst = [300,300,300]
    #model = model_run.create_full_dense_model("standard",neuron_lst)
    
    model_run.evaluate_standard(run_type="standard",
                                num_runs=run_cnt, 
                                final_acc = f_acc,
                                es_delta = 0.1,
                                log_handler = log_handler)
    
    std_model_name = log_handler.get_modelname()
    model_run.save_model(std_model_name)
    
    prune_filename = utils.add_time_to_filename("prune_details")
    log_handler.set_log_filename(prune_filename)
    log_handler.write_to_file(prune_filename)
    del model_run

#train_standard()


def train_standard_with_regularization():
    log_dir = "standard_with_regularization"
    model_run = mr.ModelRun(db)
    
    log_handler = utils.LogHandler(log_dir, tensorboard_dir, prune_dir, 
                                   model_dir, plot_dir)
    
    model_name = "standard_with_regularization"
    model_name = utils.add_time_to_filename(model_name)
    log_handler.set_log_filename(model_name)
    
    #neuron_lst = [300,300,300]
    #model = model_run.create_full_dense_model("standard",neuron_lst)
    
    model_run.evaluate_standard_with_regularizer(run_type="standard_with_regularization",
                                num_runs=run_cnt, 
                                final_acc = f_acc,
                                es_delta = 0.1,
                                log_handler = log_handler)
    
    std_model_name = log_handler.get_modelname()
    model_run.save_model(std_model_name)
    
    prune_filename = utils.add_time_to_filename("prune_details")
    log_handler.set_log_filename(prune_filename)
    log_handler.write_to_file(prune_filename)
    del model_run

#train_standard_with_regularization()


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
    
    #neuron_update_lst = ["ctr","act","act_acc"]
    prune_type_lst = ["neuron","neuron_wts"]
    reset_neuron_lst = [False]
    
    for r_neuron in reset_neuron_lst:
        for n_pruning in n_pruning_lst:
                for p_type in prune_type_lst:
                    for p_pct in prune_pct_lst:
                        model_name = "cip" 
                        #model_name += "_PruneStart_" + str(prune_start_at)
                        model_name += "_NumPruning_" + str(n_pruning)
                        model_name += "_PrunePct_" + str(p_pct)
                        #model_name += "_FinalAcc_" + str(f_acc)
                        #model_name += "_NeuronUpdate_" + n_update
                        model_name += "_PruningType_" + p_type
                        if r_neuron == True:
                            model_name += "_ResetNeuron"
                        model_name += "_EarlyStopping_" + str(delta)
                        
                        model_name = utils.add_time_to_filename(model_name)
                        
                        log_handler.set_log_filename(model_name)
                        
                        #model_run.set_log_handler(log_handler)
                        
                        model_run.evaluate_cip(run_type="cip", 
                                               prune_start_at_acc = prune_start_at,
                                               num_pruning = n_pruning,
                                               final_acc = f_acc,
                                               prune_pct = p_pct,
                                               neuron_update = "ctr",
                                               pruning_type = p_type,
                                               reset_neuron = r_neuron,
                                               early_stopping_delta=delta,
                                               log_handler = log_handler)
                        
                        log_handler.log_single_run(model_name)
                        
                        prune_model_name = log_handler.get_modelname()
                        model_run.save_model(prune_model_name)
                        
    prune_filename = utils.add_time_to_filename("prune_details")
    log_handler.set_log_filename(prune_filename)
    log_handler.write_to_file(prune_filename)
    del model_run

#cip_pruning()

def prune_standard_tf(trained_model_dir,
                      tf_pruned_model_dir):
    trained_model = keras.models.load_model(trained_model_dir)    
    
    initial_sparsity=0
    final_sparsity=0.80
    tensorboard_dir = tf_pruned_model_dir + "/tb_results"
    tf_prune = tfp.TFPrune(db, tensorboard_dir)
    (history, test_result, tf_pruned_model) = tf_prune.prune_dense(trained_model, initial_sparsity, final_sparsity)
    
    model_dir = tf_pruned_model_dir + "/model"
    tf_pruned_model.save(model_dir)
    filename = tf_pruned_model_dir + "/tf_prune_history.xlsx"
    tf_prune.log(history, test_result, filename)

#std_dir = root_log_dir + "/standard/model/standard_2022_08_15-23_23_02"
#tf_pruned_model_dir = root_log_dir + "/tf_std_pruned"
#prune_standard_tf(std_dir, tf_pruned_model_dir)


def prune_trained(trained_model_dir):    
    
    log_dir = "prune_trained"
    
    r_type = "prune_trained"
    prune_start_at = 50/100
    n_pruning = 5
    f_acc = 91/100
    p_pct = 80
    r_neuron = False
    delta = 0.1
    
    # neuron_update: ctr,act,act_acc
    n_update = "ctr"
    # pruning_type: neuron, neuron_wts
    p_type = "neuron_wts"
    
    #neuron_update_list = ["ctr","act","act_acc"]
    #prune_type_list = ["neuron","neuron_wts"]
    
    model_run = mr.ModelRun(db)
    log_handler = utils.LogHandler(log_dir, tensorboard_dir, prune_dir, 
                               model_dir, plot_dir)
    
    model_name = r_type
    model_name += "_NumPruning_" + str(n_pruning)
    model_name += "_NeuronUpdate_" + n_update
    model_name += "_PruningType_" + p_type
    if r_neuron == True:
        model_name += "_ResetNeuron"
    model_name += "_EarlyStopping_" + str(delta)
    
    model_name = utils.add_time_to_filename(model_name)
    
    log_handler.set_log_filename(model_name)
    
    trained_model = keras.models.load_model(trained_model_dir)
    
    
    model_run.prune_trained_model(run_type=r_type,
                                  prune_start_at_acc = prune_start_at,
                                  trained_model=trained_model,
                                  num_pruning=n_pruning,
                                  final_acc=f_acc,
                                  prune_pct=p_pct,
                                  neuron_update = n_update,
                                  pruning_type = p_type,
                                  reset_neuron = r_neuron,
                                  early_stopping_delta=delta,
                                  log_handler = log_handler)
    
    prune_model_name = log_handler.get_modelname()
    model_run.save_model(prune_model_name)
    
    prune_filename = utils.add_time_to_filename("prune_details")
    log_handler.set_log_filename(prune_filename)
    log_handler.write_to_file(prune_filename)
    del trained_model

#trained_model_dir = "./standard/model/standard_2022_08_15-23_23_02"    
#prune_trained( trained_model_dir )




def matrix_norms(std_dir):
    std_model = keras.models.load_model(std_dir)
    #tf_model = keras.models.load_model(tf_dir)
    matrix_norm_dir = "./train_pruned/matrix_norm/"
    if not os.path.exists(matrix_norm_dir):
        os.makedirs(matrix_norm_dir)

    for trained_model_dir in glob.glob("./train_pruned/model/*"):
        norm_file = matrix_norm_dir + trained_model_dir.split("/")[-1] + ".xlsx"        
        trained_model = keras.models.load_model(trained_model_dir)    
        df = utils.generate_matrix_norms(std_model, trained_model)    
        df.to_excel(norm_file)      


std_dir = model_dir + "/standard"
#freq_dir = model_dir + "/cip_NumPruning_2_PruningInterval_1_FinalAcc_0.92_TotalPrune_80"
tf_dir = model_dir + "/tf_pruned"

#matrix_norms(std_dir)


def matrix_heatmap():    
    matrix_heatmap_dir = "./train_pruned/matrix_heatmap/"
    if not os.path.exists(matrix_heatmap_dir):
        os.makedirs(matrix_heatmap_dir)
    
    svd_plots = gp.SVDPlots()
    
    std_dir = "./standard/model/standard_2022_08_15-23_23_02"
    freq_dir = "./prune_trained/model/prune_trained_NumPruning_5_NeuronUpdate_ctr_PruningType_neuron_wts_EarlyStopping_0.1_2022_08_19-12_02_11"
    tf_dir = "./tf_prune_trained/model"        
    
    prefix = "pruned"
    gp.matrix_heatmap( std_dir, freq_dir, tf_dir, matrix_heatmap_dir, prefix)
    svd_plots.ConvertToEps(matrix_heatmap_dir)


    std_dir = "./standard/model/standard_2022_08_15-23_23_02"
    freq_dir = "./train_pruned/model/freq_train_pruned_2022_08_19-13_50_26"
    tf_dir = "./train_pruned/model/tf_train_pruned_2022_08_19-13_44_17"        
    
    prefix = "trained"
    gp.matrix_heatmap( std_dir, freq_dir, tf_dir, matrix_heatmap_dir, prefix)
    svd_plots.ConvertToEps(matrix_heatmap_dir)

#matrix_heatmap()

def rewind_and_train(model_dir_name):
    log_dir = "rewind_train"
    
    r_type = "rewind_train"
    f_acc = 98/100
    es_delta = 0.1
    
    model_run = mr.ModelRun(db)
    log_handler = utils.LogHandler(log_dir, tensorboard_dir, prune_dir, 
                               model_dir, plot_dir)
    
    pruned_model = keras.models.load_model(model_dir_name)
    
    log_file_name = "rewind_train"
    log_file_name = utils.add_time_to_filename(log_file_name)
    log_handler.set_log_filename(log_file_name)
    
    model_run.rewind_and_train( run_type = r_type,
                                model=pruned_model,
                                final_acc=f_acc,
                                early_stopping_delta = es_delta,
                                log_handler = log_handler)
    
    model_name = log_handler.get_modelname()
    model_run.save_model(model_name)

model_dir_name = root_log_dir + "/cip/model/cip_NumPruning_5_PrunePct_80_NeuronUpdate_ctr_PruningType_neuron_wts_EarlyStopping_0.1_2022_08_06-02_55_38"
#rewind_and_train(model_dir_name)

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


def cnn_pruning():
    """
    Evaluates pruning after reaching specific training accuracy 
    for each stage of pruning    

    Returns
    -------
    None.

    """
    #data.Data.load_disk = classmethod(data.Data.load_disk)
    #data.Data.load_disk()
    
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
    
    neuron_update = "act"
    prune_type = "neuron"
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
    
    model_name = utils.add_time_to_filename(model_name)
    
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
    
    prune_filename = utils.add_time_to_filename("prune_details")
    log_handler.set_log_filename(prune_filename)
    log_handler.write_to_file(prune_filename)
    del model_run

#cnn_pruning()

def prune_fully_dense(nw_type):
    """
    Evaluates pruning after reaching specific training accuracy 
    for each stage of pruning    

    Returns
    -------
    None.

    """
    log_dir = nw_type
    model_run = mr.ModelRun(db)
    
    log_handler = utils.LogHandler(log_dir, tensorboard_dir, prune_dir, 
                               model_dir, plot_dir)
    
    prune_start_at = 80/100
    p_pct = 70
    n_pruning = 2
    f_acc = 92/100
    #r_neuron = False
    delta = 0.1
    
    # neuron_update: ctr,act,act_acc
    #n_update = "act"
    # pruning_type: neuron, neuron_wts
    #p_type = "neuron"
    
    neuron_update_type = "ctr"#,"act","act_acc"]
    prune_type = "neuron_wts"#["neuron","neuron_wts"]
    r_neuron = False
    
        
    model_name = nw_type 
    #model_name += "_PruneStart_" + str(prune_start_at)
    model_name += "_NumPruning_" + str(n_pruning)
    model_name += "_PrunePct_" + str(p_pct)
    #model_name += "_FinalAcc_" + str(f_acc)
    model_name += "_NeuronUpdate_" + neuron_update_type
    model_name += "_PruningType_" + prune_type
    if r_neuron == True:
        model_name += "_ResetNeuron"
    model_name += "_EarlyStopping_" + str(delta)
    
    model_name = utils.add_time_to_filename(model_name)
    
    log_handler.set_log_filename(model_name)
    
    #model_run.set_log_handler(log_handler)
    
    if nw_type == "prune_fully_dense_network":
        num_layers = 4
        num_neurons = 300
        model_run.evaluate_fully_dense(run_type=nw_type, 
                               prune_start_at_acc = prune_start_at,
                               num_pruning = n_pruning,
                               final_acc = f_acc,
                               prune_pct = p_pct,
                               neuron_update = neuron_update_type,
                               pruning_type = prune_type,
                               reset_neuron = r_neuron,
                               early_stopping_delta=delta,
                               log_handler = log_handler,
                               num_layers = num_layers,
                               num_neurons = num_neurons)
    elif nw_type == "cip":
        model_run.evaluate_cip(run_type=nw_type, 
                               prune_start_at_acc = prune_start_at,
                               num_pruning = n_pruning,
                               final_acc = f_acc,
                               prune_pct = p_pct,
                               neuron_update = neuron_update_type,
                               pruning_type = prune_type,
                               reset_neuron = r_neuron,
                               early_stopping_delta=delta,
                               log_handler = log_handler)
        
    log_handler.log_single_run(model_name)
    
    prune_model_name = log_handler.get_modelname()
    model_run.save_model(prune_model_name)
            
    prune_filename = utils.add_time_to_filename("prune_details")
    log_handler.set_log_filename(prune_filename)
    log_handler.write_to_file(prune_filename)
    del model_run

#nw_type = "prune_fully_dense_network"
#nw_type = "cip"
#prune_network(nw_type)

