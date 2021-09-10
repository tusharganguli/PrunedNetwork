#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 19:14:33 2021

@author: tushar
"""

import tensorflow as tf
from tensorflow import keras
from keras.models import Model

model_name = "mnist_model.h5"

# Create a custom layer to add the state variables for counting the frequency 
# of each neuron activation
class MyDense(keras.layers.Layer):
    def __init__(self,units,activation=None, **kwargs):
        super(MyDense,self).__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        print("self.units:"+ str(self.units))
        
    def build(self, input_shape):
        # initialize the weights in this layer
        self.kernel = self.add_weight(name="kernel",shape=[input_shape[-1],self.units],
                                      initializer="glorot_normal", trainable=True)
        #initialize the bias
        self.bias = self.add_weight(name="bias",shape=[self.units], initializer="zeros")
        self.add_weight(name="activation_result", shape=[self.units], 
                                           initializer="zeros",trainable=False)
        
        self.neuron_freq = self.add_weight(name="neuron_freq", shape=[self.units], 
                                           initializer="zeros",trainable=False)
        super().build(input_shape)
    
    def call(self, X):
        activation_result = self.activation(tf.matmul(X,self.kernel) + self.bias)
        #weights = self.get_weights()
        tf.print("Activation Result")
        tf.print(activation_result)
        self.add_metric(activation_result,"activation_result")
        #self.neuron_freq = self.neuron_freq + 1
        #self.neuron_freq = tf.where(activation_result>0,self.neuron_freq+1,self.neuron_freq)
        return activation_result
    
    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape.as_list()[:-1]+ [self.units])
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units":self.units, 
                "activation": keras.activations.serialize(self.activation)}
                #"neuron_freq": self.neuron_freq}
    
    def __update_neuron(self, activation_result):
        self.neuron_freq = tf.where(activation_result>0,self.neuron_freq+1,self.neuron_freq)
        

       
def create_and_store_model(train_images, train_labels ):
    
    # Define the model architecture.
    input_layer = keras.Input(shape=(28,28), name="input")
    flatten = keras.layers.Flatten(name="flatten")(input_layer)
    mydense_1 = MyDense(100,activation=tf.nn.relu, name="mydense" )(flatten)
    output_layer = keras.layers.Dense(10, activation=tf.nn.softmax, name="output")(mydense_1)
    model = keras.models.Model(inputs=input_layer,outputs=output_layer)
    
    # Train the digit classification model
    # after a model is created, call the compile function to specify the 
    # loss function and optimizer to use.
    model.compile(optimizer=keras.optimizers.SGD(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[keras.metrics.SparseCategoricalAccuracy()])
    
    #output_layers = ['output1', 'output2']
    #model.metrics_tensors += [layer.output for layer in model.layers if layer.name in output_layers]
    #model.metrics_names += [layer.output for layer in model.layers ]
    
    #from tensorflow.keras import backend as K
    #import datetime
    
    class MyCallback(keras.callbacks.Callback):
        def __init__(self,inputs):
            self.inputs = inputs
            
        def on_train_batch_end(self,batch, logs=None):
            hidden_layer = None
            input_layer = None
            for layer in model.layers:
                if layer.name in ["mydense"]:
                    hidden_layer = layer
                    tf.print(logs["activation_result"])
                elif layer.name in ["input"]:
                    input_layer = layer
            #inp = model.input
            #op = hidden_layer.output
            #functor = K.function([inp], [op])    
            #print(functor([tf.reshape(hidden_layer.data,[28,28])]))
        """
        def on_train_batch_end(self,batch, logs=None):
            layer_names = [layer.name for layer in model.layers]
            
            # input 
            inp = model.input
            # all layer outputs            
            outputs = [layer.output for layer in model.layers]
            # evaluation functions
            functors = [K.function([inp], [out]) for out in outputs]    

            # Testing
            count = 0
            for func in functors:
              print('\n')
              print("Layer Name: ",layer_names[count])
              print('\n')
              print(func([batch]))
              count+=1
  
        """
        """
        def on_train_batch_end(self, batch, logs=None): 
            for layer in model.layers:
                if isinstance(layer, keras.layers.InputLayer) == True:
                    inputs = layer.input
                if isinstance(layer, keras.layers.Flatten) == False and\
                    isinstance(layer, keras.layers.InputLayer) == False:
                    get_layer_output = K.function(inputs=inputs, outputs = layer.output) 
                    tf.print('\n Training: output of the layer {} is {} ends at {}'.format(layer, get_layer_output.outputs , datetime.datetime.now().time()))   
        """
        """
        def on_train_batch_end(self, batch, logs):
            #layer1 = model.layers[1].get_weights()
            for layer in model.layers:
                if isinstance(layer, keras.layers.Flatten) == False:
                    intermediate_layer_model = Model(inputs=model.input,
                                                     outputs=layer.output)
                    intermediate_output = intermediate_layer_model.predict(self.inputs)
                    #print(layer.neuron_freq)
                    #config = layer.get_config()
                    #print(config["activation"])
                    #weights = layer.get_weights()
                    #activation_result = layer.activation(tf.matmul(layer.input,weights[0]) + weights[1])
                    #print(activation_result)
                    #tf.where(activation_result>0,weights[2]+1,weights[2])
                    #layer.set_weights(weights)
        """
        
    history = model.fit(train_images, train_labels, epochs=1, validation_split=0.1,
                        callbacks=[MyCallback(train_images)])
    # Evaluate baseline test accuracy and save the model for later usage.
        
    model.save(model_name)
    
    
def evaluate_model(model, test_images, test_labels ):
    _, model_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print('Model accuracy:', model_accuracy)
    
def load_model():
    return keras.models.load_model(model_name)


class Linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        self.w = self.add_weight(
            shape=(input_dim, units), initializer="random_normal", trainable=True
        )
        self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
    






