#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 09:36:03 2023

@author: tushar
"""

from tensorflow import keras
import numpy as np
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import data

pca_dir = "/home/tushar/datadrive/Spyder/NetworkPruning/PrunedNetwork/LogDir/pca"
pca_model_dir = pca_dir + "/models"
lrm_dir = pca_dir + "/low_rank_models/"

def get_rank(model,df):
    
    for layer in model.layers:
        if not isinstance(layer,keras.layers.Dense) or layer.name == "output":
            continue
        wts = layer.weights
        rank = np.linalg.matrix_rank(wts[0])    
        df[layer.name] = rank
    return df

def eval_pca(std_wts,row):
    
    low_rank_wts = []
    idx = 0
    for idx in range(0,len(row)):
        # Apply PCA to find the principal components
        rank = int(row[idx])
        pca = PCA(n_components=rank)
        pca.fit(std_wts[idx])
        # Transform the data using the selected principal components
        approximation = pca.transform(std_wts[idx])
        # Reconstruct the matrix using the reduced number of principal components
        reconstructed_wts = pca.inverse_transform(approximation)
        low_rank_wts.append(reconstructed_wts)
    return low_rank_wts

def evaluate_and_save_models(df, std_model, model, low_rank_wts, 
                             test_img, test_labels, model_name):
    model_name = lrm_dir + model_name
    idx = 0
    
    for layer in std_model.layers:
        if not isinstance(layer, keras.layers.Dense) or layer.name == "output":
            continue
        wts = low_rank_wts[idx]
        idx += 1
        layer_wts = layer.get_weights()
        layer_wts[0] = wts
        layer.set_weights(layer_wts)
    #std_model.save(model_name)
    eval_result1 = std_model.evaluate(test_img,test_labels)
    eval_result1 = [x*100 for x in eval_result1]
    eval_result2 = model.evaluate(test_img,test_labels)
    eval_result2 = [x*100 for x in eval_result2]
    
    
    #df["Standard Low Rank Model-Loss"] = eval_result[0] * 100
    #df["Model-Loss"] = eval_result[0]
    df["Standard Low Rank Model-Accuracy"] = eval_result1[1]
    df["Model-Accuracy"] = eval_result2[1]
    df["Standard Low Rank Model-Top 1% Accuracy"] = eval_result1[2]
    df["Model-Top 1% Accuracy"] = eval_result2[2]
    df["Standard Low Rank Model-Top 5% Accuracy"] = eval_result1[3]
    df["Model-Top 5% Accuracy"] = eval_result2[3]
    
    return df
    
def evaluate_lowrank(database):
    
    # get the weghts of the standard model
    std_dir = pca_model_dir + "/standard_2022_08_15-23_23_02"
    std_model = keras.models.load_model(std_dir)
    std_wts = [layer.get_weights()[0] for layer in std_model.layers if isinstance(layer, keras.layers.Dense) and layer.name != "output"]
    
    cwd = os.getcwd()
    os.chdir(pca_model_dir)
    df = pd.DataFrame(columns=["Model"])
    
    data_obj = data.Data(database)
    (train_img,valid_img,test_img,
     train_labels,valid_labels,test_labels) = data_obj.load_data()
    
    for m_dir in glob.glob("*"):
        df_new = pd.DataFrame([[m_dir]],columns=["Model"])
        model = keras.models.load_model(m_dir)
        df_new = get_rank(model,df_new)
        low_rank_wts = eval_pca(std_wts,df_new.iloc[-1][1:])
        df_new = evaluate_and_save_models(df_new, std_model, model, 
                                 low_rank_wts, test_img, test_labels, m_dir)
        df = pd.concat([df,df_new])
    os.chdir(cwd)
    return df
    

filename = pca_dir + "/pca_result.xlsx"
#df = evaluate_lowrank()
#df.to_excel(filename, index=False)

        

def perform_pca(wts):
    # performing standardization
    sc = StandardScaler()
    wts_scaled = sc.fit_transform(wts)

    pca = PCA(n_components=None)
    #for wts in std_wts:
    pca.fit(wts_scaled)
    return pca

def scree_plot(pca):
    # plot a scree plot
    components = len(pca.explained_variance_ratio_)
    plt.plot(range(1,components+1), np.cumsum(pca.explained_variance_ratio_ * 100))
    plt.xlabel("Number of components")
    plt.ylabel("Explained variance (%)") 
           
    
def pca_analysis():
    
    std_dir = pca_model_dir + "/standard_2022_08_15-23_23_02"
    std_model = keras.models.load_model(std_dir)
    std_wts = [layer.get_weights()[0] for layer in std_model.layers if isinstance(layer, keras.layers.Dense) and layer.name != "output"]
    
    pca = perform_pca(std_wts[0])
    scree_plot(pca)
    
#pca_analysis()
