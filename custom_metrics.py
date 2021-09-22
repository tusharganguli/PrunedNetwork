#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 06:41:41 2021

@author: tushar
"""

import tensorflow as tf

class CustomMetrics(tf.keras.metrics.Metric):
  def __init__(self, name=None, dtype=None, **kwargs):
    super(CustomMetrics, self).__init__(name, dtype, **kwargs)
    self.count = tf.Variable(0)
    
  def update_state(self, y_true, y_pred, sample_weight=None):
    first_tensor = tf.nest.flatten(y_true)[0]
    batch_size = tf.shape(first_tensor)[0]
    self.count.assign(batch_size)

  def result(self):
    return tf.identity(self.count)