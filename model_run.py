#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 19:16:49 2021

@author: tushar
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

import data
import pruning_callback as pc
import custom_model as cmod



class CustomLossWithMessages(tf.keras.losses.Loss):
    def __init__(self, model, name='custom_loss_with_messages', **kwargs):
        self.regularize = False
        self.model = model
        super().__init__(name=name,**kwargs)
        
    def call(self, y_true, y_pred):
        main_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        # Conditionally apply the custom regularizer at specific training steps
        if self.regularize == True:
            # Calculate the regularizer term for each layer and sum them up
            regularization_loss = sum(self.singular_value_regularizer(layer.kernel) for layer in self.model.layers if isinstance(layer, tf.keras.layers.Dense))
            #regularization_loss = 0
            # Combine the regularization loss with the main loss (categorical cross-entropy)
            lambda_value = 1
            main_loss += lambda_value * regularization_loss
            #main_loss = 1000
        return main_loss
     
    def get_config(self):
        config = {
            'regularize': self.regularize
        }
        base_config = super().get_config()
        return {**base_config, **config}   
    
    # Define your custom regularizer function
    def singular_value_regularizer(self,weight_matrix):
        # Calculate the singular values of the weight matrix
        singular_values = tf.linalg.svd(weight_matrix, compute_uv=False)
        #tf.print(singular_values.shape[0])
        start_value = 1
        end_value = singular_values.shape[0]
        step = 1
        weights = tf.range(start_value, end_value+step, step, dtype=tf.float32)
        weighted_sv = tf.multiply(singular_values,weights)
        # Return the sum of singular values as the regularization term
        return tf.reduce_sum(weighted_sv)


class ModelRun():
    
    def __init__(self,data_set):
        
        # Load dataset
        #self.data_set = data_set
        data_obj = data.Data(data_set)
        (self.train_img,self.valid_img,self.test_img,
         self.train_labels,self.valid_labels,self.test_labels) = data_obj.load_data()
        
        #self.df = self.__create_data_frame()
        #self.log_dir = log_dir + "/" 
        #self.log_dir = os.path.join(os.curdir,self.log_dir)
        
        #self.prune_dir = prune_dir + "/" 
        #self.prune_dir = os.path.join(os.curdir,self.prune_dir)
        
        # hyperparameters
        self.optimizer = "sgd"
        self.loss = "sparse_categorical_crossentropy"
        self.acc_metrics = "accuracy"
        self.top1_metrics = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top1')
        self.top5_metrics = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5')
        
        # for creating the model
        self.epochs = 500
        
        
    def __del__(self):
        #del self.model
        #del self.data_set
        del self.train_img
        del self.valid_img
        del self.test_img
        
        
    def save_model(self, model_dir):
        """
        Saves the current model on disk.

        Parameters
        ----------
        model_dir : String
            Specifies the location where to save the model parameters.

        Returns
        -------
        None.

        """
        self.model.save(model_dir)
    
    def get_modelname(self):
        return self.lh.get_modelname()
    
    
    def set_log_handler(self, log_handler):
        self.lh = log_handler
        
    def evaluate_standard(self, run_type, 
                          num_runs, final_acc, 
                          es_delta, log_handler):
        
        
        tf.keras.backend.clear_session()
        model = self.create_model(run_type)
        
        stop_cb = pc.StopCallback(final_acc)
        es_cb = pc.EarlyStoppingCallback(delta=es_delta, 
                                         verbose=1)
        
        tb_log_filename = log_handler.get_tensorboard_dir()
        tensorboard_cb = keras.callbacks.TensorBoard(tb_log_filename)
        
        model.compile(optimizer=self.optimizer, loss=self.loss, 
                      metrics=[self.acc_metrics,self.top1_metrics, 
                               self.top5_metrics])
        
        
        history = model.fit(self.train_img, self.train_labels, 
                            epochs=self.epochs,
                            validation_data=(self.valid_img,self.valid_labels),
                            callbacks=[es_cb, stop_cb,tensorboard_cb])
        
        self.model = model
        
        num_epochs = stop_cb.get_num_epochs()
        eval_result = model.evaluate(self.test_img,self.test_labels)
        
        del model
        log_handler.log_data(run_type, history, eval_result, 
                         num_epochs )
    
    
    def evaluate_standard_with_regularizer(self, run_type, 
                          num_runs, final_acc, 
                          es_delta, log_handler):
        
        
        tf.keras.backend.clear_session()
        model = self.create_regularizer_model()
        
        stop_cb = pc.StopCallback(final_acc)
        es_cb = pc.EarlyStoppingCallback(delta=es_delta, 
                                         verbose=1)
        
        tb_log_filename = log_handler.get_tensorboard_dir()
        tensorboard_cb = keras.callbacks.TensorBoard(tb_log_filename)
        
        # Create an instance of the custom loss class
        self.loss_function = CustomLossWithMessages(model)
    
        reg_cb = pc.RegularizerCallback(self.loss_function)
        
        
        model.compile(optimizer=self.optimizer, loss=self.loss_function, 
                      metrics=[self.acc_metrics,self.top1_metrics, 
                               self.top5_metrics], run_eagerly=True)
        
        
        history = model.fit(self.train_img, self.train_labels, 
                            epochs=self.epochs,
                            validation_data=(self.valid_img,self.valid_labels),
                            callbacks=[es_cb, stop_cb,tensorboard_cb, reg_cb])
        
        self.model = model
        
        num_epochs = stop_cb.get_num_epochs()
        eval_result = model.evaluate(self.test_img,self.test_labels)
        
        del model
        log_handler.log_data(run_type, history, eval_result, 
                         num_epochs )

    
    # Define a function to conditionally apply the regularizer based on training steps
    def custom_loss_with_arg(self, model):
        def custom_loss(y_true, y_pred):
            # Conditionally apply the custom regularizer at specific training steps
            if pc.regularizer == True:
                pc.verify_flag = True
                # Calculate the regularizer term for each layer and sum them up
                regularization_loss = sum(self.singular_value_regularizer(layer.kernel) for layer in model.layers if isinstance(layer, tf.keras.layers.Dense))
                #regularization_loss = 0
                # Combine the regularization loss with the main loss (categorical cross-entropy)
                main_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
                lambda_value = 1
                total_loss = main_loss + lambda_value * regularization_loss
                return total_loss
            else:
                return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        return custom_loss

    def create_regularizer_model(self):
        
        
        input_layer = keras.Input(shape=(28,28), name="input")
        flatten = keras.layers.Flatten(name="flatten")(input_layer)
        
        dense_1 = keras.layers.Dense(300,activation=tf.nn.relu, 
                                     name="dense_1")(flatten)
        dense_2 = keras.layers.Dense(200,activation=tf.nn.relu, 
                                     name="dense_2")(dense_1)
        dense_3 = keras.layers.Dense(100,activation=tf.nn.relu, 
                                     name="dense_3")(dense_2)
        dense_4 = keras.layers.Dense(50,activation=tf.nn.relu, 
                                     name="dense_4")(dense_3)
        #dense_5 = keras.layers.Dense(50,activation=tf.nn.relu, 
        #                             name="dense_5" )(dense_4)
        output_layer = keras.layers.Dense(10, activation=tf.nn.softmax, 
                                          name="output")(dense_4)
        
        model = keras.models.Model(inputs=input_layer,outputs=output_layer)
        return model
    
    def evaluate_cnn(self, run_type, 
                     prune_start_at_acc,
                     num_pruning,
                     final_acc,
                     prune_pct,
                     neuron_update,
                     pruning_type,
                     reset_neuron,
                     early_stopping_delta,
                     log_handler):
        
        tf.keras.backend.clear_session()
        model = self.create_model(run_type)
        
        stop_cb = pc.StopCallback(final_acc)
        
        tb_log_filename = log_handler.get_tensorboard_dir()
        tensorboard_cb = keras.callbacks.TensorBoard(tb_log_filename)
        
        pruning_cb = pc.CNNCallback(model,
                                    self.train_img,
                                    prune_start_at_acc,
                                    num_pruning,    final_acc,
                                    prune_pct,      neuron_update,
                                    pruning_type,   reset_neuron,
                                    log_handler, tb_log_filename
                                    )
        
        # simple early stopping
        #es_cb = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
        es_cb = pc.EarlyStoppingCallback(delta=early_stopping_delta, 
                                         verbose=1)
        #prune_model_name = self.get_modelname() #+ ".h5"
        #mc_cb = ModelCheckpoint(prune_model_name, monitor='val_accuracy', 
        #                        mode='max', verbose=1, save_best_only=True)
        
        model.compile(optimizer=self.optimizer, 
                      loss=self.loss,
                      metrics=[self.acc_metrics, self.top1_metrics, self.top5_metrics], 
                      run_eagerly=True
                      )

        history = model.fit(self.train_img, 
                            self.train_labels, 
                            epochs=self.epochs,
                            validation_data=(self.valid_img,self.valid_labels),
                            callbacks=[es_cb, stop_cb, 
                                      pruning_cb, tensorboard_cb
                                      ]
                            )
        
        self.model = model
        
        num_epochs = stop_cb.get_num_epochs()
        eval_result = model.evaluate(self.test_img,self.test_labels)
        
        (total_trainable_wts,
        total_pruned_wts,prune_pct_achieved) = self.__generate_model_summary(model)
        prune_pct_achieved = prune_pct_achieved.numpy()
        del model
        
        log_handler.log_data(run_type, history, eval_result, 
                         num_epochs, prune_pct_achieved )

    def evaluate_layer_wise(self, run_type, prune_start_at_acc,
                                  final_acc, prune_pct,
                                  neuron_update, pruning_type,
                                  reset_neuron, early_stopping_delta,
                                  log_handler):
        
        tf.keras.backend.clear_session()
        model = self.create_model(run_type)
        
        
        pruning_cb = pc.LayerPruneCallback(model, self.train_img,
                                           prune_start_at_acc, final_acc,
                                           prune_pct, neuron_update,
                                           pruning_type, reset_neuron,
                                           log_handler
                                           )
        
        model.compile(optimizer=self.optimizer, 
                      loss=self.loss,
                      metrics=[self.acc_metrics,self.top1_metrics, 
                               self.top5_metrics], run_eagerly=True)
        
        stop_cb = pc.StopCallback(final_acc)
        
        tb_log_filename = log_handler.get_tensorboard_dir()
        tensorboard_cb = keras.callbacks.TensorBoard(tb_log_filename)
        
        # simple early stopping
        #es_cb = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
        es_cb = pc.EarlyStoppingCallback(delta=early_stopping_delta, 
                                         verbose=1)
        #prune_model_name = self.get_modelname() #+ ".h5"
        #mc_cb = ModelCheckpoint(prune_model_name, monitor='val_accuracy', 
        #                        mode='max', verbose=1, save_best_only=True)
        
        history = model.fit(self.train_img, 
                            self.train_labels, 
                            epochs=self.epochs,
                            validation_data=(self.valid_img,self.valid_labels),
                            callbacks=[es_cb, stop_cb, 
                                       pruning_cb, tensorboard_cb])
        
        self.model = model
        
        num_epochs = stop_cb.get_num_epochs()
        eval_result = model.evaluate(self.test_img,self.test_labels)
        
        (total_trainable_wts,
        total_pruned_wts,prune_pct_achieved) = self.__generate_model_summary(model)
        prune_pct_achieved = prune_pct_achieved.numpy()
        del model
        
        log_handler.log_data(run_type, history, eval_result, 
                         num_epochs, prune_pct_achieved )

    def evaluate_cip(self, run_type, 
                     prune_start_at_acc,
                     num_pruning,
                     final_acc,
                     prune_pct,
                     neuron_update,
                     pruning_type,
                     reset_neuron,
                     early_stopping_delta,
                     log_handler):
        
        tf.keras.backend.clear_session()
        model = self.create_model(run_type)
        
        
        pruning_cb = pc.CIPCallback(model,
                                    self.train_img,
                                    prune_start_at_acc,
                                    num_pruning,    final_acc,
                                    prune_pct,      neuron_update,
                                    pruning_type,   reset_neuron,
                                    log_handler
                                    )
        
        model.compile(optimizer=self.optimizer, 
                      loss=self.loss,
                      metrics=[self.acc_metrics,self.top1_metrics, 
                               self.top5_metrics], 
                      run_eagerly=True)
        
        stop_cb = pc.StopCallback(final_acc)
        
        tb_log_filename = log_handler.get_tensorboard_dir()
        tensorboard_cb = keras.callbacks.TensorBoard(tb_log_filename)
        
        # simple early stopping
        #es_cb = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
        es_cb = pc.EarlyStoppingCallback(delta=early_stopping_delta, 
                                         verbose=1)
        #prune_model_name = self.get_modelname() #+ ".h5"
        #mc_cb = ModelCheckpoint(prune_model_name, monitor='val_accuracy', 
        #                        mode='max', verbose=1, save_best_only=True)
        
        history = model.fit(self.train_img, 
                            self.train_labels, 
                            epochs=self.epochs,
                            validation_data=(self.valid_img,self.valid_labels),
                            callbacks=[es_cb, stop_cb, 
                                       pruning_cb, tensorboard_cb])
        
        self.model = model
        
        num_epochs = stop_cb.get_num_epochs()
        eval_result = model.evaluate(self.test_img,self.test_labels)
        
        (total_trainable_wts,
        total_pruned_wts,prune_pct_achieved) = self.__generate_model_summary(model)
        prune_pct_achieved = prune_pct_achieved.numpy()
        del model
        
        log_handler.log_data(run_type, history, eval_result, 
                         num_epochs, prune_pct_achieved )

    
    def evaluate_fully_dense(self, run_type, 
                     prune_start_at_acc,
                     num_pruning,
                     final_acc,
                     prune_pct,
                     neuron_update,
                     pruning_type,
                     reset_neuron,
                     early_stopping_delta,
                     log_handler,
                     num_layers,
                     num_neurons):
        
        tf.keras.backend.clear_session()
        model = self.create_fully_dense_model(run_type, num_layers, num_neurons)
        
        
        pruning_cb = pc.CIPCallback(model,
                                    self.train_img,
                                    prune_start_at_acc,
                                    num_pruning,    final_acc,
                                    prune_pct,      neuron_update,
                                    pruning_type,   reset_neuron,
                                    log_handler
                                    )
        
        model.compile(optimizer=self.optimizer, 
                      loss=self.loss,
                      metrics=[self.acc_metrics,self.top1_metrics, 
                               self.top5_metrics], 
                      run_eagerly=True)
        
        stop_cb = pc.StopCallback(final_acc)
        
        tb_log_filename = log_handler.get_tensorboard_dir()
        tensorboard_cb = keras.callbacks.TensorBoard(tb_log_filename)
        
        # simple early stopping
        #es_cb = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
        es_cb = pc.EarlyStoppingCallback(delta=early_stopping_delta, 
                                         verbose=1)
        #prune_model_name = self.get_modelname() #+ ".h5"
        #mc_cb = ModelCheckpoint(prune_model_name, monitor='val_accuracy', 
        #                        mode='max', verbose=1, save_best_only=True)
        
        history = model.fit(self.train_img, 
                            self.train_labels, 
                            epochs=self.epochs,
                            validation_data=(self.valid_img,self.valid_labels),
                            callbacks=[es_cb, stop_cb, 
                                       pruning_cb, tensorboard_cb])
        
        self.model = model
        
        num_epochs = stop_cb.get_num_epochs()
        eval_result = model.evaluate(self.test_img,self.test_labels)
        
        (total_trainable_wts,
        total_pruned_wts,prune_pct_achieved) = self.__generate_model_summary(model)
        prune_pct_achieved = prune_pct_achieved.numpy()
        del model
        
        log_handler.log_data(run_type, history, eval_result, 
                         num_epochs, prune_pct_achieved )

    def prune_trained_model(self, run_type, prune_start_at_acc, 
                            trained_model, num_pruning, final_acc, 
                            prune_pct, neuron_update, pruning_type, 
                            reset_neuron, early_stopping_delta, 
                            log_handler):
        
        tf.keras.backend.clear_session()
        
        input_layer = trained_model.input
        output_layer = trained_model.output
            
        model = cmod.CustomModel(inputs=input_layer,outputs=output_layer)
        
        model.compile(optimizer=self.optimizer, 
                      loss=self.loss,
                      metrics=[self.acc_metrics,self.top1_metrics, 
                               self.top5_metrics], 
                      run_eagerly=True)
        
        #stop_cb = pc.StopCallback(final_acc)
        
        tb_log_filename = log_handler.get_tensorboard_dir()
        tensorboard_cb = keras.callbacks.TensorBoard(tb_log_filename)
        
        pruning_cb = pc.CIPCallback(model,
                                    prune_start_at_acc,
                                    num_pruning,    final_acc,
                                    prune_pct,      neuron_update,
                                    pruning_type,   reset_neuron,
                                    log_handler
                                    )
        
        es_cb = pc.EarlyStoppingCallback(delta=early_stopping_delta, 
                                         verbose=1)
        
        history = model.fit(self.train_img, 
                            self.train_labels, 
                            epochs=self.epochs,
                            validation_data=(self.valid_img,self.valid_labels),
                            callbacks=[es_cb, pruning_cb, tensorboard_cb])
        
        self.model = model
        
        num_epochs = pruning_cb.get_num_epochs()
        
        eval_result = model.evaluate(self.test_img,self.test_labels)
        (total_trainable_wts,
        total_pruned_wts,prune_pct_achieved) = self.__generate_model_summary(model)
        prune_pct_achieved = prune_pct_achieved.numpy()
        del model
        
        log_handler.log_data(run_type, history, eval_result, 
                         num_epochs, prune_pct_achieved )

            
    def train_pruned_model(self, run_type, pruned_model, 
                           final_training_acc, early_stopping_delta,
                           log_handler):
        
            
        tf.keras.backend.clear_session()
        cloned_model = keras.models.clone_model(pruned_model)
        cloned_model.set_weights(pruned_model.get_weights())
        
        input_layer = cloned_model.input
        output_layer = cloned_model.output
            
        model = keras.models.Model(inputs=input_layer,outputs=output_layer)
        
        stop_cb = pc.StopCallback(final_training_acc)
        
        tb_log_filename = log_handler.get_tensorboard_dir()
        tensorboard_cb = keras.callbacks.TensorBoard(tb_log_filename)
        
        es_cb = pc.EarlyStoppingCallback(delta=early_stopping_delta, 
                                         verbose=1)
        
        model.compile(optimizer=self.optimizer, 
                      loss=self.loss,
                      metrics=[self.acc_metrics,self.top1_metrics, 
                               self.top5_metrics], 
                      run_eagerly=True)

        history = model.fit(self.train_img, 
                            self.train_labels, 
                            epochs=self.epochs,
                            validation_data=(self.valid_img,self.valid_labels),
                            callbacks=[es_cb, stop_cb, tensorboard_cb])
        
        self.model = model
        
        
        num_epochs = stop_cb.get_num_epochs()
        
        eval_result = model.evaluate(self.test_img,self.test_labels)
                                                             
        (total_trainable_wts,
        total_pruned_wts,prune_pct_achieved) = self.__generate_model_summary(model)
        prune_pct_achieved = prune_pct_achieved.numpy()
        del model
        del cloned_model
        
        log_handler.log_data(run_type, history, eval_result, 
                             num_epochs, prune_pct_achieved )
    
    def rewind_and_train(self, run_type, model, final_acc, 
                         early_stopping_delta, log_handler):
        
        tf.keras.backend.clear_session()
        #cloned_model = keras.models.clone_model(model)
        #cloned_model.set_weights(model.get_weights())
        
        for layer in model.layers:
            if not isinstance(layer,keras.layers.Dense):
                continue
            wts = layer.get_weights()
            rand_mat = np.random.randn(wts[0].shape[0],wts[0].shape[1])
            wts[0][np.where(wts[0] != 0)] = rand_mat[np.where(wts[0] != 0)]
            layer.set_weights(wts)

        input_layer = model.input
        output_layer = model.output
            
        model = cmod.CustomModel(inputs=input_layer,outputs=output_layer)
        model.disable_neuron_update()
        model.preserve_pruning()
       
        model.compile(optimizer=self.optimizer, 
                      loss=self.loss,
                      metrics=[self.acc_metrics,self.top1_metrics, 
                               self.top5_metrics], 
                      run_eagerly=True)
        
        tb_log_filename = log_handler.get_tensorboard_dir()
        tensorboard_cb = keras.callbacks.TensorBoard(tb_log_filename)
        
        stop_cb = pc.StopCallback(final_acc)
        
        es_cb = pc.EarlyStoppingCallback(delta=early_stopping_delta, 
                                         verbose=1)
        
        history = model.fit(self.train_img, 
                            self.train_labels, 
                            epochs=self.epochs,
                            validation_data=(self.valid_img,self.valid_labels),
                            callbacks=[es_cb, tensorboard_cb])
        
        self.model = model
        
        num_epochs = stop_cb.get_num_epochs()
        
        eval_result = model.evaluate(self.test_img,self.test_labels)
        (total_trainable_wts,
        total_pruned_wts,prune_pct_achieved) = self.__generate_model_summary(model)
        prune_pct_achieved = prune_pct_achieved.numpy()
        del model
        
        log_handler.log_data(run_type, history, eval_result, 
                         num_epochs, prune_pct_achieved )

        
    def evaluate_optimal_pruning(self, run_type, num_runs, pruning_type,
                                 final_training_accuracy, 
                                 epoch_pruning_interval, num_pruning,  
                                 reset_neuron_count = False ):
        
        history_list = []
        evaluate_list = []
        epoch_list = []
        prune_pct_list = []
        
        log_file_name = run_type
        log_file_name += "_PruneType_" + pruning_type
        log_file_name += "_EpochInterval_" + str(epoch_pruning_interval)
        log_file_name += "_TotalPruning_" + str(num_pruning)
        log_file_name += "_FinalAcc_" + str(final_training_accuracy)
        
        if reset_neuron_count == True:
            log_file_name += "_ResetNeuron"    
            
        for runs in range(num_runs):   
            tf.keras.backend.clear_session()
            model = self.__create_model(run_type)
            
            stop_cb = pc.StopCallback(final_training_accuracy)
            
            run_log_dir = self.__get_run_logdir(self.log_dir,log_file_name)
            tensorboard_cb = keras.callbacks.TensorBoard(run_log_dir)
            pruning_cb = pc.OptimalPruningCallback(model,pruning_type,
                                                   epoch_pruning_interval,
                                                   num_pruning, 
                                                   reset_neuron_count,
                                                   self.log_dir,
                                                   self.prune_dir, log_file_name)
        
            model.compile(optimizer=self.optimizer, 
                          loss=self.loss,
                          metrics=[self.metrics], 
                          run_eagerly=True)

            history = model.fit(self.train_img, 
                                self.train_labels, 
                                epochs=self.epochs,
                                validation_data=(self.valid_img,self.valid_labels),
                                callbacks=[stop_cb,pruning_cb,tensorboard_cb])
            
            self.model = model
            
            num_epochs = stop_cb.get_num_epochs()
            epoch_list.append(num_epochs)
            
            history_list.append(history)    
            eval_result = model.evaluate(self.test_img,self.test_labels)
            evaluate_list.append(eval_result)
            (train_loss,train_accuracy, 
             val_loss, val_accuracy,
             test_loss, test_accuracy) = self.__generate_avg(history_list, 
                                                         evaluate_list)
                                                                 
            (total_trainable_wts,
            total_pruned_wts,prune_pct_achieved) = self.__generate_model_summary(model)
            prune_pct_achieved = prune_pct_achieved.numpy()
            prune_pct_list.append(prune_pct_achieved)
            del model
        
        self.__log_data(run_type, history_list, evaluate_list, epoch_list,
                        prune_pct_list )

        
    def __generate_model_summary(self,model):
        trainable_wts_cnt = 0
        trainable_wts = model.trainable_weights
        pruned_wts = 0
        for wts in trainable_wts:
            if "kernel" in wts.name:
                trainable_wts_cnt = tf.add(trainable_wts_cnt,tf.size(wts))
                kernel_arr = wts.numpy()
                num_zeros = tf.size(kernel_arr[np.where(kernel_arr == 0)])
                pruned_wts = tf.add(pruned_wts,num_zeros)
                
        tf.print("Trainable variables:",trainable_wts_cnt)
        tf.print("Variables pruned:",pruned_wts)
        prune_pct = (pruned_wts/trainable_wts_cnt)*100
        tf.print("Prune percentage:",prune_pct)
        return (trainable_wts,pruned_wts,prune_pct)
    
    def create_basic_cnn(self):
        from functools import partial
        
        DefaultConv2D = partial(keras.layers.Conv2D,
                            kernel_size=3, activation='relu', padding="SAME")
        
        input_layer = keras.Input(shape=[28, 28, 1])
        conv_1 = DefaultConv2D(filters=32, kernel_size=3)(input_layer)
        max_pool_1 = keras.layers.MaxPooling2D(pool_size=2)(conv_1)
        conv_2 = DefaultConv2D(filters=64)(max_pool_1)
        max_pool_2 = keras.layers.MaxPooling2D(pool_size=2)(conv_2)
        flatten = keras.layers.Flatten()(max_pool_2)
        dense_2 = keras.layers.Dense(units=64, activation='relu')(flatten)
        output_layer = keras.layers.Dense(units=10, activation='softmax', 
                                    name='output')(dense_2)
        
        model = cmod.CustomModel(inputs=input_layer,outputs=output_layer)
        return model


    def create_cnn(self):
        from functools import partial
        
        DefaultConv2D = partial(keras.layers.Conv2D,
                            kernel_size=3, activation='relu', padding="SAME")
        
        input_layer = keras.Input(shape=[28, 28, 1])
        conv_1 = DefaultConv2D(filters=64, kernel_size=7)(input_layer)
        max_pool_1 = keras.layers.MaxPooling2D(pool_size=2)(conv_1)
        conv_2 = DefaultConv2D(filters=128)(max_pool_1)
        conv_3 = DefaultConv2D(filters=128)(conv_2)
        max_pool_2 = keras.layers.MaxPooling2D(pool_size=2)(conv_3)
        conv_4 = DefaultConv2D(filters=256)(max_pool_2)
        conv_5 = DefaultConv2D(filters=256)(conv_4)
        max_pool_3 = keras.layers.MaxPooling2D(pool_size=2)(conv_5)
        flatten = keras.layers.Flatten()(max_pool_3)
        dense_1 = keras.layers.Dense(units=128, activation='relu')(flatten)
        #keras.layers.Dropout(0.5),
        dense_2 = keras.layers.Dense(units=64, activation='relu')(dense_1)
        #keras.layers.Dropout(0.5),
        output_layer = keras.layers.Dense(units=10, activation='softmax', 
                                    name='output')(dense_2)
        
        model = cmod.CustomModel(inputs=input_layer,outputs=output_layer)
        return model
    
    def create_inception_v3(self, run_type):
        iv3 = keras.applications.inception_v3.InceptionV3(include_top=True, weights=None,
                                                    classes=10)
        
        if run_type != "standard":
            model = cmod.CustomModel(inputs=iv3.input,outputs=iv3.output)
        else:
            model = iv3
        return model
    
    def create_model(self,run_type):
        if run_type == "cnn":
            #return self.create_basic_cnn()
            #return self.create_cnn()
            return self.create_inception_v3(run_type)
        
        input_layer = keras.Input(shape=(28,28), name="input")
        flatten = keras.layers.Flatten(name="flatten")(input_layer)
        
        dense_1 = keras.layers.Dense(300,activation=tf.nn.relu, 
                                     name="dense_1" )(flatten)
        dense_2 = keras.layers.Dense(200,activation=tf.nn.relu, 
                                     name="dense_2" )(dense_1)
        dense_3 = keras.layers.Dense(100,activation=tf.nn.relu, 
                                     name="dense_3" )(dense_2)
        dense_4 = keras.layers.Dense(50,activation=tf.nn.relu, 
                                     name="dense_4" )(dense_3)
        #dense_5 = keras.layers.Dense(50,activation=tf.nn.relu, 
        #                             name="dense_5" )(dense_4)
        output_layer = keras.layers.Dense(10, activation=tf.nn.softmax, 
                                          name="output")(dense_4)
        
        if run_type != "standard":
            model = cmod.CustomModel(inputs=input_layer,outputs=output_layer)
        else:
            model = keras.models.Model(inputs=input_layer,outputs=output_layer)
        return model
    
    
    def create_full_dense_model(self,run_type, neuron_lst):
        num_layers = len(neuron_lst)
        if num_layers > 4:
            raise ValueError("Maximum 4 layers supported")
            
        input_layer = keras.Input(shape=(28,28), name="input")
        flatten = keras.layers.Flatten(name="flatten")(input_layer)
        
        dense_1 = keras.layers.Dense(neuron_lst[0],activation=tf.nn.relu, name="dense_1" )(flatten)
        final_dense = dense_1
        if num_layers >= 2:
            dense_2 = keras.layers.Dense(neuron_lst[1],activation=tf.nn.relu, name="dense_2" )(dense_1)
            final_dense = dense_2
        if num_layers >= 3:
            dense_3 = keras.layers.Dense(neuron_lst[2],activation=tf.nn.relu, name="dense_3" )(dense_2)
            final_dense = dense_3
        if num_layers >= 4:
            dense_4 = keras.layers.Dense(neuron_lst[3],activation=tf.nn.relu, name="dense_4" )(dense_3)
            final_dense = dense_4
        
        output_layer = keras.layers.Dense(10, activation=tf.nn.softmax, name="output")(final_dense)
        if run_type != "standard":
            model = cmod.CustomModel(inputs=input_layer,outputs=output_layer)
        else:
            model = keras.models.Model(inputs=input_layer,outputs=output_layer)
        return model
        
    def create_fully_dense_model(self,run_type, num_layers, num_neurons=300):
        
        input_layer = keras.Input(shape=(28,28), name="input")
        flatten = keras.layers.Flatten(name="flatten")(input_layer)
        prev_layer = flatten
        for idx in range(num_layers):
            layer_name = "dense_" + str(idx)
            curr_layer = keras.layers.Dense(num_neurons,activation=tf.nn.relu, 
                                     name=layer_name )(prev_layer)
            prev_layer = curr_layer
            
        output_layer = keras.layers.Dense(10, activation=tf.nn.softmax, 
                                          name="output")(curr_layer)
        
        if run_type != "standard":
            model = cmod.CustomModel(inputs=input_layer,outputs=output_layer)
        else:
            model = keras.models.Model(inputs=input_layer,outputs=output_layer)
        return model
        
        
        
        
        
        
        
        
        
        
        