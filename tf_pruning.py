#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 08:04:34 2022

@author: tushar
"""

import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import numpy as np
import tempfile
import pandas as pd
import data
import os

class TFPrune:
    def __init__(self, db, tensorboard_dir):
        """
        

        Parameters
        ----------
        db : TYPE
            DESCRIPTION.
        tensorboard_dir : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        tensorboard_dir += "/" 
        self.tensorboard_dir = os.path.join(os.curdir,tensorboard_dir)
        
        self.prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        
        # Compute end step to finish pruning after 2 epochs.
        self.batch_size = 32
        self.epochs = 2
        self.validation_split = 0.1 # 10% of training set will be used for validation set. 

        data_obj = data.Data(db, validation_split=0)
        (self.train_img,self.valid_img,self.test_img,
         self.train_labels,self.valid_labels,self.test_labels) = data_obj.load_data()

        num_images = self.train_img.shape[0] * (1 - self.validation_split)
        self.end_step = np.ceil(num_images / self.batch_size).astype(np.int32) * self.epochs


    def prune(self, model, i_sparsity, f_sparsity):
        """
        Runs the pruning on the passed trained model

        Parameters
        ----------
        model : Keras model instance.
            Trained model which needs to be pruned.
        i_sparsity : Int
            Initial sparsity.
        f_sparsity : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # Define model for pruning.
        pruning_params = {
              'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=i_sparsity,
                                                                       final_sparsity=f_sparsity,
                                                                       begin_step=0,
                                                                       end_step=self.end_step)
        }

        
        
        model_for_pruning = self.prune_low_magnitude(model, **pruning_params)

        # `prune_low_magnitude` requires a recompile.
        model_for_pruning.compile(optimizer='sgd',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        #model_for_pruning.summary()

        logdir = tempfile.mkdtemp()

        callbacks = [
          tfmot.sparsity.keras.UpdatePruningStep(),
          tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
        ]
        
        model_for_pruning.fit(self.train_img, self.train_labels,
                          batch_size=self.batch_size, epochs=self.epochs, 
                          validation_split=self.validation_split,
                          callbacks=callbacks)
        
        _, model_for_pruning_accuracy = model_for_pruning.evaluate(
           self.test_img, self.test_labels, verbose=0)
        
        #print('Baseline test accuracy:', baseline_model_accuracy) 
        print('Pruned test accuracy:', model_for_pruning_accuracy)

    def prune_dense(self, model, i_sparsity, f_sparsity):
        # Define model for pruning.
        pruning_params = {
              'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=i_sparsity,
                                                                       final_sparsity=f_sparsity,
                                                                       begin_step=0,
                                                                       end_step=self.end_step)
        }

        # Helper function uses `prune_low_magnitude` to make only the 
        # Dense layers train with pruning.
        def apply_pruning_to_dense(layer):
            if isinstance(layer, tf.keras.layers.Dense) and layer.name != 'output':
                return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
            return layer
        
        
        # Use `tf.keras.models.clone_model` to apply `apply_pruning_to_dense` 
        # to the layers of the model.
        model_for_pruning = tf.keras.models.clone_model(
            model,
            clone_function=apply_pruning_to_dense,
        )
        
        #model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model_for_pruning, 
        #                                                             **pruning_params)
        #model.summary()
        model_for_pruning.summary()
        
        # `prune_low_magnitude` requires a recompile.
        model_for_pruning.compile(optimizer='sgd',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        #model_for_pruning.summary()

        callbacks = [
          tfmot.sparsity.keras.UpdatePruningStep(),
          tfmot.sparsity.keras.PruningSummaries(log_dir=self.tensorboard_dir),
        ]
        
        history = model_for_pruning.fit(self.train_img, self.train_labels,
                          epochs=self.epochs, 
                          validation_split=self.validation_split,
                          callbacks=callbacks)
        
        test_result = model_for_pruning.evaluate(
           self.test_img, self.test_labels, verbose=0)
        
        #print('Pruned test loss:', test_result[0])
        print('Pruned test accuracy:', test_result[1])
        
        model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
        return (history, test_result, model_for_export)
    
    def log( self, history, test_result, filename):
        df = pd.DataFrame(columns=["Training Accuracy", 
                                   "Validation Accuracy",
                                   "Test Accuracy",
                                   "Training Loss", 
                                   "Validation Loss",
                                   "Test Loss"])
        
        train_loss = history.history["loss"][-1]
        train_accuracy = history.history["accuracy"][-1]
        val_loss = history.history["val_loss"][-1]
        val_accuracy = history.history["val_accuracy"][-1]
        
        test_loss = test_result[0]
        test_accuracy = test_result[1]
        
        log_data = [train_accuracy, val_accuracy, test_accuracy,
                    train_loss, val_loss, test_loss]
        
        df2 = pd.DataFrame([log_data], columns=list(df))
        
        df2.to_excel(filename)