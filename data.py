#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 02:10:47 2021

@author: tushar
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import os

class Data():
    def __init__(self,dataset,validation_split=0.1):
        self.dataset = dataset
        self.validation_split = validation_split
    
    def load_disk(self):
        """
        Loads the database from the disk locally

        Returns
        -------
        None.

        """
        #data_dir = "/home/tushar/datadrive/Spyder/NetworkPruning/PrunedNetwork/Imagenet"
        data_dir = "../Imagenet"
        #write_dir = data_dir# + "/temp"
        write_dir = "../temp"
        download_config = tfds.download.DownloadConfig(
                            extract_dir=os.path.join(write_dir, 'extracted'),
                            manual_dir=data_dir
                            ) 
        """
        download_and_prepare_kwargs = {
                        'download_dir': os.path.join(write_dir, 'downloaded'),
                        'download_config': download_config,
                        }
        ds = tfds.load('imagenet2012', 
               data_dir=os.path.join(write_dir, 'data'),         
               split='train', 
               shuffle_files=False, 
               download=False, 
               as_supervised=True,
               download_and_prepare_kwargs=download_and_prepare_kwargs)
        """
        builder = tfds.builder("imagenet2012")
        builder.download_and_prepare(download_config=download_config)
        ds = builder.as_dataset(split=self.validation_split)
        return ds
        
    def load_data(self):
        if type(self.dataset) != type(tf.keras.datasets.fashion_mnist):
            ds = self.load_disk()
            (train_img, train_labels), (test_img, test_labels) = ds.load_data()
        else:
            # Load MNIST dataset
            (train_img, train_labels), (test_img, test_labels) = self.dataset.load_data()
        validation_sz = tf.cast(train_img.shape[0] * self.validation_split, dtype=tf.int32)
        # Normalize the input image so that each pixel value is between 0 and 1.
        # create validation data set
        valid_img, train_img = train_img[:validation_sz] / 255.0,\
                                train_img[validation_sz:] / 255.0
        valid_labels, train_labels = train_labels[:validation_sz],\
                                    train_labels[validation_sz:]
        test_img = test_img / 255.0
        return (train_img,valid_img,test_img,train_labels,valid_labels,test_labels)