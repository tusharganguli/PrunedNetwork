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

##############################################################################

# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 and 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

def create_model():
    #input_layer = keras.Input(shape=(28,28), name="input")
    #flatten = keras.layers.Flatten(name="flatten")(input_layer)
    #mydense_1 = cl.MyDense(100,activation=tf.nn.relu, name="dense" )(flatten)
    #output_layer = keras.layers.Dense(10, activation=tf.nn.softmax, name="output")(mydense_1)
    #model = keras.models.Model(inputs=input_layer,outputs=output_layer)
    model = cm.CustomModel()
    return model

tf.keras.backend.clear_session()
model = create_model()
model.compile(optimizer="adam", loss="mse", metrics=["mae"], run_eagerly=True)
weight_print = cc.MyCallback(40)
model.fit(train_images, train_labels, epochs=3, callbacks=[weight_print])

#cn.create_and_store_model(train_images, train_labels)
#ct.custom_training(model,train_images,train_labels)
#model = cn.load_model()
#cn.evaluate_model(model, test_images, test_labels)

