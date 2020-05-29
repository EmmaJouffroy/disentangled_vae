# -*- coding: utf-8 -*-

####
# JOUFFROY Emma stagiaire 2020
# https://yann-leguilly.gitlab.io/post/2019-10-09-tensorflow-tfdata/ 
####

from glob import glob
import tensorflow as tf
import IPython.display as display
from PIL import Image 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, pathlib, pickle



AUTOTUNE = tf.data.experimental.AUTOTUNE


class DataSet():
    """
    This class creates the dataset using a keras ImageDataGenerator object.
    It allows to generate in real time shifted and zoomed image from originals. 
    """
    def __init__(self, dir_path, batch_size):
        """
        df : dataframe which contains two columns (path of images, labels)
        dir_path : data_dir as a path object
        batch_size : number of images in each batch
        dir_pkl : directory to a pkl file
        class_names : a numpy array containing every class of the object, corresponding 
                    to the name of this object (the name of the folder containing the image)
        img_height, img_width : 128 pixel (the image is squared)
        pkl_dict : a list which will contained all decoded pkl file
        """
        self.data_dir = dir_path
        self.batch_size = batch_size
        self.dir_path = pathlib.Path(dir_path)
        self.dir_pkl = glob(self.data_dir+"*/*.pkl")
        self.class_names = np.array([item.name for item in self.dir_path.glob('*') if item.name != ".DS_Store"])
        self.img_height, self.img_width = 128, 128
        self.pkl_dict = []

    def create_pkl_dict(self):
        """
        Function which reads all pkls from a list of path
        and append it in an empty dictionay in order to be
        readable
        """
        # dir_pkl contains every path_file of pkl file
        for pkl_file in self.dir_pkl:
            # we open the pkl_file
            with open(pkl_file, 'rb') as f:
                # load every data
                data = pickle.load(f)
                # append it in the array
                self.pkl_dict.append([pkl_file, data])

    
    def get_name_object(self, file_path):
        """
        Get all the labels (filename, scalex, scaley scalez and quaternions) 
        associated with the image

        file_path: path of the image
        """
        # if the pkl_dict is empy, then we need to create it
        if len(self.pkl_dict) == 0:
            self.create_pkl_dict()
        # we instantiate an empty dictionary
        labels_dict = {}
        # we get an array of splitted filepath
        parts = file_path.split(os.path.sep)
        # we get the name of the file
        file_name = (parts[-2] == self.class_names)
        array_class_name = [f for f in file_name]
        # we get the number of the frame corresponding to the image
        name_frame = parts[-1].split('.')[0]
        number_frame = name_frame.split('_')[-1]
        number_frame_tensor = int(number_frame)
        # we initalize values for labels
        scale_x, scale_y, scale_z, quaternion = 0., 0., 0., np.ones((1,4))
        # for every path contained ni pkl_dict
        for n in range(len(self.pkl_dict)):
            # we split the path to obtain the name of the corresponding frame
            pkl_split = self.pkl_dict[n][0].split(os.path.sep)
            # if the name of the frame contained in the pkl_dict is equal
            # to the name of the frame of the filepath
            if(parts[-2] == pkl_split[-2]):
                # the we get the x_scale value on the pkl_dict data
                scale_x = self.pkl_dict[n][1].get('scale_x')
                # the we get the y_scale value on the pkl_dict data
                scale_y = self.pkl_dict[n][1].get('scale_y')
                # the we get the z_scale value on the pkl_dict data
                scale_z = self.pkl_dict[n][1].get('scale_z')
                # the we get the quaternion value on the pkl_dict data corresponding to the
                # number_frame_tensor th quaternion in the pkl_dict
                quaternion = self.pkl_dict[n][1].get('quaters')[number_frame_tensor]
                # the quaternion is a string object that we need to convert in a float number
                quaternion = [ float(s) for s in quaternion ]
        # we finally update the dictionary created above with the values of the filepath and each label
        labels_dict.update( {'filename' : array_class_name, 'scale_x' : scale_x, 'scale_y' : scale_y, 'scale_z' : scale_z, 'quaternion' : quaternion})
        return file_path, str(labels_dict)

    def create_csv(self, name_csv):
        """
        name_csv : path where the csv file will be store

        Function which creates a csv file which contains path to each image
        and the corresponding labels, and returns a pandas dataframe with the same
        informations.
        """
        # we get every filepath of each image contained in the dataset folder
        filepath = [f for f in glob(str(self.data_dir+'*/*.png'), recursive=True)]
        # for each filepath we get the filepath and the corresponding labels
        # (the name of the object, the quaternion and the dimensions)
        images_map = map(self.get_name_object, filepath)
        # we transform the map object into and iterable list
        images_list = list(images_map)
        # we create a dataframe from the list created above with two columns :
        # the first for the filepath and the second for the labels
        df = pd.DataFrame(images_list, columns=["Path", "Label"])
        # we save the dataframe into a csv file
        df.to_csv(name_csv,index=False)
        return df

    def create_generator(self, width_shift, heigh_shift, zoom, val_split):
        """
        width_shift : range from where a random number will be picked for a shift on the width
        height_shift : range from where a random number will be picked for a shift on the height
        zoom : range from where a random number will be picked for a zoom
        val_split : proportion of testing data from the whole dataset

        Function which returns an ImageDataGenerator keras object, allowing to 
        process different actions (such as shift and zoom) in real time on the images.
        """
        train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            # we rescale every pixel of the image for them to be between
            # 0 and 1 (the original values are between 0 and 255)
            rescale=1./255,
            # we don't need horizontal flip nor vertical fleep
            horizontal_flip=False,
            vertical_flip=False,
            # we want to shift randomly our images to the width and height
            width_shift_range=width_shift, 
            height_shift_range=heigh_shift,
            # we want to randomly zoom our images
            zoom_range=zoom,
            # we want to split our dataset into training and validation set
            # we want to fill the pixel in black for interpolation
            # as we have black background and other interpolation would be useless
            # and even destroy the form of our object
            validation_split=val_split,
            fill_mode="constant")
        return train_gen

    def create_flow(self, generator, subset, df):
        """
        generator : a keras ImageDataGenerator object 
        subset : nor "test" either "validation" subset
        df : dataframe whith two columns, the first indicates the
        path to each image and the second the labels associated with
        the image

        Function which returns a flow of preprocessed images where
        each image is decoded from paths contained in a dataframe
        """
        flow = generator.flow_from_dataframe(
                                        # the dataset is contained in the dataframe
                                        dataframe = df,
                                        # the directory of the dataset is self.data_dir
                                        directory= self.data_dir,
                                        # we want to shuffle the dataset
                                        shuffle=True,
                                        # the size of the images to decode are (128,128)
                                        target_size=(self.img_height, self.img_width),
                                        # there is one channel on the images, so we want them
                                        # to be in grayscale
                                        color_mode='grayscale',
                                        # we need to specify the subset, or training either 
                                        # validation set
                                        subset=subset,
                                        # we want to create batch
                                        batch_size= self.batch_size,
                                        # the filepath are contained in the dataframe df's column
                                        # "Path"
                                        x_col="Path",
                                        # the labels are contained in the dataframe df's column
                                        # "Labels"
                                        y_col="Label",
                                        # we don't need the labels to be preprocessed
                                        class_mode = "raw"
                                        )
        return flow

    def show_batch(self, image_batch, label_batch):
        """
        Plots a figure of 12 images from a batch, with their name 
        associated

        image_batch: Eager tensor of all images of the batch
        label_batch: Boolean Tensor with "True" value for the name of the associated image
        """
        fig = plt.figure(figsize=(10,10))
        fig.suptitle('Batch of training set', fontsize=16)

        for n in range(12):
            ax = plt.subplot(4,4,n+1)
            # we plot the 12 first image of the batch, without considering the channel
            plt.imshow(image_batch[n, :, :, 0], cmap='gray')
            # we need to decode from bytes to utf_8 the labels, as they are encoded in the csv_file
            decoded_batch = label_batch[n].numpy().decode('utf8')
            decoded_batch = eval(decoded_batch)
            # the value of the title is the value of the index of self.CLASS_NAMES where label_batch = True
            plt.title(self.class_names[np.array(decoded_batch['filename'])==1][0].title())
            plt.axis('off')
        plt.show()


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
        # the dataset is prefetch
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds
