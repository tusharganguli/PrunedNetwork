#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:54:36 2021

@author: tushar
"""
from tensorflow import keras
import pandas as pd
import tf_pruning as tfp

import model_run as mr
import generate_plots as gp

db = keras.datasets.fashion_mnist
run_cnt = 1
# number of epochs after which pruning occurs
prune_freq = 15
t_acc = 0.98
model_dir = "model"

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

std_full_dense_dir = model_dir + "/standard_full_dense"

def train_standard():
    tensorboard_dir = "standard_results"
    prune_dir = "standard_details"
    model_run = mr.ModelRun(db,tensorboard_dir, prune_dir)
    
    neuron_lst = [300,300,300]
    model = model_run.create_full_dense_model("standard",neuron_lst)
    
    model_run.evaluate_standard(run_type="standard",
                                model=model,
                                num_runs=run_cnt, 
                                final_training_accuracy = t_acc)
    
    model_run.write_to_file(filename = "Results_Standard.xls")
    #u,sv,vt =  model.get_svd()
    #model.write_sv(u,"standard_u")
    #model.write_sv(sv,"standard_sv")
    #model.write_sv(vt,"standard_vt")
    
    #del u
    #del sv
    #del vt
    
    model_run.save_model(std_full_dense_dir)
    del model_run

#train_standard()

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
Evaluates constant pruning at regular intervals
"""

def cip_pruning():
    tensorboard_dir = "cip_prune_results"
    prune_dir = "cip_prune_details"
    model = mr.ModelRun(db,tensorboard_dir, prune_dir)
    
    prune_start_at = 85/100
    n_pruning = 2
    final_acc = 92/100
    
    model.evaluate_CIP(run_type="cip", 
                       num_runs=run_cnt, 
                       pruning_type="weights",
                       neuron_update_at_acc=1,
                       prune_start_at_acc = prune_start_at,
                       num_pruning = n_pruning,
                       final_training_acc = final_acc,
                       target_prune_pct=80)
    
    model.write_to_file(filename = "Results_CIPPruning.xls")
    
    #u,sv,vt =  model.get_svd()
    #model.write_sv(u,"cip_u")
    #model.write_sv(sv,"cip_sv")
    #model.write_sv(vt,"cip_vt")
    
    #del u
    #del sv
    #del vt
    model_name = "/cip" + "_PruneStartAt_" + str(prune_start_at)
    model_name += "_NumPruning_" + str(n_pruning)
    model_name += "_FinalAcc_" + str(final_acc)
    
    cip_model_dir = model_dir + model_name
    model.save_model(cip_model_dir)
    
    del model




def prune_standard_tf(trained_model_dir=model_dir + "/standard",
                      tf_pruned_model_dir = model_dir + "/tf_pruned"):
    trained_model = keras.models.load_model(trained_model_dir)    
    
    initial_sparsity=0
    final_sparsity=0.80
    tensorboard_dir = "tf_prune_results"
    tf_prune = tfp.TFPrune(db, tensorboard_dir)
    tf_pruned_model = tf_prune.prune_dense(trained_model, initial_sparsity, final_sparsity)
    
    tf_pruned_model.save(tf_pruned_model_dir)

tf_pruned_model_dir = model_dir + "/tf_pruned_dense"
#prune_standard_tf(std_full_dense_dir, tf_pruned_model_dir)


def prune_standard_freq_based(trained_model_dir = model_dir + "/standard"):
    trained_model = keras.models.load_model(trained_model_dir)    
    
    tensorboard_dir = "cip_prune_results"
    prune_dir = "cip_prune_details"
    model_run = mr.ModelRun(db,tensorboard_dir, prune_dir)
    n_pruning = 2
    p_int = 1
    final_acc = 92/100
    prune_pct = 80
    
    model_run.prune_trained_model(run_type="trained",
                                  trained_model=trained_model,
                                  num_pruning=n_pruning,
                                  pruning_interval=p_int,
                                  final_training_acc=final_acc,
                                  total_prune_pct=prune_pct)
    
    model_name = "/cip" + "_NumPruning_" + str(n_pruning)
    model_name += "_PruningInterval_" + str(p_int)
    model_name += "_FinalAcc_" + str(final_acc)
    model_name += "_TotalPrune_" + str(prune_pct)
    
    cip_model_dir = model_dir + model_name
    model_run.save_model(cip_model_dir)

#prune_standard_freq_based(std_full_dense_dir)

def train_pruned_model(freq_model_dir, tf_model_dir, 
                       freq_trained_dir, tf_trained_dir,
                       tf_log_filename, freq_log_filename, num_runs):
    freq_pruned_model = keras.models.load_model(freq_model_dir)    
    tf_pruned_model = keras.models.load_model(tf_model_dir)    
    
    tensorboard_dir = "train_pruned_results"
    prune_dir = "train_pruned_details"
    model_run = mr.ModelRun(db,tensorboard_dir, prune_dir)
    r_type="train_pruned"
    final_acc = 0.98
    
    #for idx in range(num_runs):
        #model_run.train_pruned_model(r_type,tf_pruned_model, final_acc)
        #model_run.save_model(tf_trained_dir)
    #model_run.write_to_file(filename = tf_log_filename)
        
    for idx in range(num_runs):
        model_run.train_pruned_model(r_type,freq_pruned_model, final_acc)
        #model_run.save_model(freq_trained_dir)
    model_run.write_to_file(filename = freq_log_filename)
    

freq_pruned_dir = model_dir + "/cip_NumPruning_2_PruningInterval_1_FinalAcc_0.92_TotalPrune_80"
tf_pruned_dir = model_dir + "/tf_pruned_dense"
freq_trained_dir = model_dir + "/train_freq_dense_pruned"
tf_trained_dir = model_dir + "/train_tf_dense_pruned"
tf_log_filename = model_dir + "/Log_TrainedPrunedModel_Tf.xls"
freq_log_filename = model_dir + "/Log_TrainedPrunedModel_Freq.xls"

num_runs = 10
train_pruned_model(freq_pruned_dir, tf_pruned_dir, 
                   freq_trained_dir, tf_trained_dir,
                   tf_log_filename, freq_log_filename, num_runs)


def matrix_norms(std_model_dir, freq_model_dir, tf_model_dir, norm_file):
    
    std_model = keras.models.load_model(std_model_dir)
    freq_model = keras.models.load_model(freq_model_dir)    
    tf_model = keras.models.load_model(tf_model_dir)
    
    df = mr.ModelRun.generate_matrix_norms(std_model, freq_model, tf_model)    
    df.to_excel(norm_file)      


std_dir = model_dir + "/standard_full_dense"
freq_dir = model_dir + "/cip_NumPruning_2_PruningInterval_1_FinalAcc_0.92_TotalPrune_80"
tf_dir = model_dir + "/tf_pruned_dense"
norm_file = model_dir+"/matrix_norm_standard_pruned.xlsx"

#matrix_norms(std_dir,freq_dir,tf_dir, norm_file)

freq_dir = model_dir + "/train_freq_dense_pruned"
tf_dir = model_dir + "/train_tf_dense_pruned"
norm_file = model_dir+"/matrix_norm_trained_pruned.xlsx"

#matrix_norms(std_dir,freq_dir,tf_dir, norm_file)
 
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

