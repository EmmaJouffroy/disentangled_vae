# -*- coding: utf-8 -*-

####
# JOUFFROY Emma stagiaire 2020
# https://www.tensorflow.org/tutorials/load_data/images
####

import os, pathlib, pickle
from glob import glob
import tensorflow as tf
import IPython.display as display
from PIL import Image 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

import tensorflow_addons as tfa

AUTOTUNE = tf.data.experimental.AUTOTUNE

"""
 The code below is used when the data augmentation is not done 
 with the function ImageDataGenerator() of keras is used : only Dataset
 of tensorflow. 
"""
class DataSetNotAugmented():
    """
    This class allows us to create a usable dataset that we will use for training
    """
    def __init__(self, dir_path, batch_size, pkl_dict):
        """
        data_dir : directory of the training dataset
        dir_path : data_dir as a path object
        class_names : names of the folder contained in the dataset
        batch_size : number of images in each batch
        img_height, img_width : size of the image
        steps_per_epoch :  number of batches to complete one epoch
        """
        self.pkl_dict = pkl_dict
        self.data_dir = dir_path
        self.dir_path = pathlib.Path(dir_path)
        self.class_names = np.array([item.name for item in self.dir_path.glob('*') if item.name != ".DS_Store"])
        self.image_count = len(list(self.dir_path.glob('*/*.png')))
        self.batch_size = batch_size
        self.img_height, self.img_width = 128, 128
        self.steps_per_epoch = np.ceil(self.image_count/self.batch_size)
    
    def show_batch(self, image_batch, label_batch):
        """
        Plots a figure of 25 images from a batch, with their labels 
        associated

        image_batch: Eager tensor of all images of the batch
        label_batch: Boolean Tensor with "True" value for the name of the associated image
        """
        fig = plt.figure(figsize=(10,10))
        fig.suptitle('Batch of training set', fontsize=16)

        for n in range(12):
            ax = plt.subplot(4,4,n+1)
            plt.imshow(image_batch[n, :, :, 0], cmap='gray')
            # the value of the title is the value of the index of self.CLASS_NAMES where label_batch = True
            decoded_batch = label_batch[n].numpy().decode('utf8')
            decoded_batch = eval(decoded_batch)
            plt.title(self.class_names[np.array(decoded_batch['filename'])==1][0].title())
            plt.axis('off')
        plt.show()
    
    def get_name_object(self, file_path):
        """
        Get the name associated with the image, 
        meaning the name of the folder from the image
        has been taken

        file_path: path of the image
        """
        #Split elements of input based on sep into a RaggedTensor.
        labels_dict = {}
        parts = file_path.split(os.path.sep)
        file_name = (parts[-2] == self.class_names)
        name_frame = parts[-1].split('.')[0]
        number_frame = name_frame.split('_')[-1]
        number_frame_tensor = int(number_frame)
        scale_x, scale_y, scale_z, quaternion = 0., 0., 0., np.ones((1,4))
        for n in range(len(self.pkl_dict)):
            pkl_split = self.pkl_dict[n][0].split(os.path.sep)
            if(parts[-2] == pkl_split[-2]):
                scale_x = self.pkl_dict[n][1].get('scale_x')
                scale_y = self.pkl_dict[n][1].get('scale_y')
                scale_z = self.pkl_dict[n][1].get('scale_z')
                quaternion = self.pkl_dict[n][1].get('quaters')[number_frame_tensor]
                quaternion = [ float(s) for s in quaternion ]
        labels_dict.update( {'filename' : file_name, 'scale_x' : scale_x, 'scale_y' : scale_y, 'scale_z' : scale_z, 'quaternion' : quaternion})
        return file_path, labels_dict

    def get_name_label(self, file_path):
        """
        Get the name associated with the image, 
        meaning the name of the folder from the image
        has been taken

        file_path: path of the image
        """
        #Split elements of input based on sep into a RaggedTensor.
        parts = tf.strings.split(file_path, os.path.sep)
        file_name = (parts[-2] == self.class_names)
        return file_name

    def decode_img(self, img):
        """
        Returns a float32 image of size (128,128,3)
        from an encoded image

        img: a tensor of type string
        """
        #Decode a PNG-encoded image to a uint8 or uint16 tensor
        img = tf.image.decode_png(img, channels=1, dtype=tf.uint8)
        #img = tf.image.rgb_to_grayscale(img)
        #Convert image to the [0,1] range
        img = tf.image.convert_image_dtype(img, tf.float32)
        #Resize images to size using the specified method.
        resized_img = tf.image.resize(img, [self.img_width, self.img_height])
        return resized_img  
    
    def process_path(self, file_path):
        """
        Returns an image object with the label associated

        file_path: path of the image as Tensor object
        """
        # Get the name object of the image
        label = self.get_name_label(file_path)
        #Reads and outputs the entire contents of the input filename.
        img = tf.io.read_file(file_path)
        #Get the returned image from the filename
        img = self.decode_img(img)
        return img, label

    def prepare_for_training(self, ds,  cache=True, shuffle_buffer_size=1000):
        """
        Returns an iterable PrefetchDataset object

        ds: a mapped dataset of images and labels
        cache: name of the file to cache the data
        shuffle_buffer_size: size of the buffer for prefetch
        """
        if cache:
            #Returns True is cache is of type string
            if isinstance(cache, str):
                # the data are cached in the file (cache) that will persist across runs.
                ds = ds.cache(cache)
            else:
                # the data are cached in memory
                ds = ds.cache()
        # randomly shuffle the element of ds, sampled from buffer_size
        #ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        # Combines consecutive elements of ds of size self.BATCH_SIZE
        #ds = ds.batch(self.batch_size, drop_remainder=True)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds
