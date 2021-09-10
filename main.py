#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:54:36 2021

@author: tushar
"""
import tensorflow as tf
from tensorflow import keras
import custom_network as cn
import custom_training as ct

##############################################################################

# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 and 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

#cn.create_and_store_model(train_images, train_labels)
ct.custom_training(train_images,train_labels)
#model = cn.load_model()
#cn.evaluate_model(model, test_images, test_labels)

"""
x = tf.ones((2, 2))
linear_layer = cn.MyDense(4, "relu")
y = linear_layer(x)
print(y)
"""