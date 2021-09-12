#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:54:36 2021

@author: tushar
"""
import tensorflow as tf
from tensorflow import keras
import custom_layer as cl
import custom_network as cn
import custom_training as ct
import custom_model as cm
import custom_callback as cc
import display as disp

##############################################################################

# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 and 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

def create_model():
    input_layer = keras.Input(shape=(28,28), name="input")
    flatten = keras.layers.Flatten(name="flatten")(input_layer)
    mydense_1 = keras.layers.Dense(100,activation=tf.nn.relu, name="dense" )(flatten)
    output_layer = keras.layers.Dense(10, activation=tf.nn.softmax, name="output")(mydense_1)
    model = keras.models.Model(inputs=input_layer,outputs=output_layer)
    return model

tf.keras.backend.clear_session()
no_of_epochs = 5

import os
root_log_dir = os.path.join(os.curdir,"my_logs")

def get_run_logdir(model_type):
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    run_id = model_type+run_id
    return os.path.join(root_log_dir,run_id)

run_log_dir = get_run_logdir("standard_")
tensorboard_cb = keras.callbacks.TensorBoard(run_log_dir)
standard_model = create_model()
standard_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"], run_eagerly=True)
standard_history = standard_model.fit(train_images, train_labels, 
                                      epochs=no_of_epochs,
                                      callbacks=[tensorboard_cb])

run_log_dir = get_run_logdir("sparse_")
tensorboard_cb = keras.callbacks.TensorBoard(run_log_dir)
custom_model = cm.CustomModel()
custom_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"], run_eagerly=True)
weight_print = cc.MyCallback(80)
custom_history = custom_model.fit(train_images, train_labels, 
                                  epochs=no_of_epochs, 
                                  callbacks=[weight_print,tensorboard_cb])


#cn.create_and_store_model(train_images, train_labels)
#ct.custom_training(model,train_images,train_labels)
#model = cn.load_model()
#cn.evaluate_model(model, test_images, test_labels)

