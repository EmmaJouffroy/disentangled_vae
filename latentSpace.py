# -*- coding: utf-8 -*-

####
# JOUFFROY Emma stagiaire 2020
# https://openreview.net/forum?id=Sy2fzU9gl
####

import tensorflow as tf
from collections import defaultdict
import numpy as np
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import glob, os, random

tfd = tfp.distributions

class LatentSpaceVisualisation():
    """
    This class implements functions that allows us
    to proove the disentanglement of the latent space of our
    beta-VAE, in a qualitative and quantitative way
    """
    def __init__(self, model, path_quali, path_quanti):
        """
        model : a trained VAE that we want to focus on
        path_quali : the relative path to the data we use for 
                    qualitative visualisation
        path_quanti : the relative path to the data we use for 
                    quantitative visualisation
        """
        self.model = model
        self.path_quali = path_quali
        self.path_quanti = path_quanti

    #########################################################################################
            # The function below are used for qualitative 
            # visualiation of latent space
    #########################################################################################

    def prepare_data(self, filepath):
        """
        filepath : 

        Function which returns a decoded image of shape (128,128,1)
        or a decoded image and the coordinates (x,y) of the center of
        the object in the image (in case of translations). 
        """
        img = tf.io.read_file(filepath)
        # decode the image into int, considering only one channel (grayscale)
        img = tf.image.decode_png(img, channels=1, dtype=tf.uint8)
        # scaling the image between 0 and 1
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resizing the image to shape (1, 128, 128)
        img = tf.image.resize(img, [128,128])
        # getting splitted element of the filepath
        parts = tf.strings.split(filepath, '/')
        parts_name = tf.strings.split(parts[-1], os.path.sep)
        # for translations, we also need the coordinates of the center 
        # of the object that are in the filepath (x,y)
        if(parts[-2]=='translations'):
            name_frame = tf.strings.split(parts[-1], '.')
            numbers = tf.strings.split(name_frame[0], '_')
            x_string = numbers[1]
            x = tf.strings.to_number(x_string, tf.int32)
            y_string = numbers[-2]
            y = tf.strings.to_number(y_string, tf.int32)
            return [img, x, y]
        else:
            return img

    def create_dataset_qualitative_visualisation(self):
        """
        Function which returns list of images for all dataset useful
        for qualitative visualisation
        For each dataset needed, we get a list of the filepath associated with every image
        contained in a foder, we then apply the function prepare_data(filepath) to every element
        of this list and then transform the resulted map object to an iterable list.
        """
        quaternion_list = [f for f in glob.glob(str(self.path_quali+'quaternion/*.png'), recursive=True)]
        quaternion_map = map(self.prepare_data, quaternion_list)
        quaternions = list(quaternion_map)
        
        typeform_list = [f for f in glob.glob(str(self.path_quali+'typeforme/*.png'), recursive=True)]
        typeform_map = map(self.prepare_data, typeform_list)
        typeform = list(typeform_map)
        
        echelleglob_list = [f for f in glob.glob(str(self.path_quali+'echelleglob/*.png'), recursive=True)]
        echelleglob_map = map(self.prepare_data, echelleglob_list)
        echelleglob = list(echelleglob_map)
        
        position_list = [f for f in glob.glob(str(self.path_quali+'translations/*.png'), recursive=True)]
        position_map = map(self.prepare_data, position_list)
        position = list(position_map)
        
        return quaternions, typeform, echelleglob, position

    def ouput_encoder(self, batch):
        """
        batch : batch (or single image) from which we need informations

        Function that encodes a batch (or a single image) and returns all the
        informations needed for the qualitative visualisation : the mean returned by the trained
        model, the variance and the mean of each activation of the latent factor along the batch, 
        the variance returned by the trained model and the sample from the distribution outputs
        by the encoder.
        """
        # we get the output distribution of the encoder, from the specified batch
        approx_posterior = self.model.encoder(batch, True)
        # we sample this distribution
        approx_posterior_sample = approx_posterior.sample()
        # we get the learned mean
        learned_mean = self.model.z_mean.numpy()[0]
        # we get the learned variance
        learned_var = self.model.z_var.numpy()[0]
        # we get the variance over the whole batch
        variance_batch = approx_posterior_sample.numpy().var(axis=0)
        # we get the mean over the whole batch
        mean_batch = approx_posterior_sample.numpy().mean(axis=0)
        return learned_mean, variance_batch, mean_batch, learned_var, approx_posterior_sample

    def plot_qualitative_function(self, dataset, index_z):
        """
        dataset : the dataset we are focused on for the visualisation 
        index_z: the factor's index of the latent variable we are
                focused on

        Function that returns an array of the mean learned by the model
        of every image in a batch
        """
        # we initiate an empty array
        mean_batchs = []
        # for every image in the considered dataset
        for data in dataset:
            # we get the learned mean of the encoder associated 
            # to this image
            mean_batch,_, _, _, _ = self.ouput_encoder(data)
            # we append the mean of the specific factor of the latent 
            # variable into the mean_batch array
            mean_batchs.append(mean_batch[index_z])
        return mean_batchs
        
    def sorted_var(self, dataset):
        """
        dataset : a batch of pertinent images used for sorting pertinent factors

        Function that returns the factors of the latent variable sorted 
        in croissant order. According to the beta-vae article, the factors
        with the smallest variances are the one that are the most disantangled. 
        """
        # we instantiate a numpy array
        variances_list = np.empty((0,32))
        # we instantate a defaultdict of type list
        sorted_variances = defaultdict(list)
        # for every image in the dataset
        for data in dataset:
            # we get the learned variance of the encoder
            _,_,_,variance,_ = self.ouput_encoder(dataset)
            # we append the resulted array 
            variances_list = np.append(variances_list, [variance], axis=0)
        # we get the mean of each variances in order to have an average 
        variances_list = variances_list.mean(axis=0)
        # we split the obtained list in order to get the average variance
        # of each factor of the latent space
        variances = np.array_split(variances_list,len(variances_list))
        # we append the defaultdict by associating the number of the 
        # factor with the value of it's avergaed variance
        for i, var in enumerate(variances):
            sorted_variances[i].append(var)
        # we finally sort the defaultdict by ordering the variances
        sorted_var = sorted(sorted_variances.items(), key=lambda x: x[1])
        return sorted_var

    def position_function(self, dataset, index_z):
        """
        dataset : the dataset of every translated images. As we use the dataset
                corresponding to the translations, we have:
                - data[0] : an image
                - data[1] : the y position of the center of the object in the image
                - data[2] : the x position of the center of the object in the image
        index_z: the factor's index of the latent variable we are
                focused on

        Function that returns a 128x128 array where each value (x,y) corresponds
        to the factor's mean activation of the image where the center of the object
        present in the image in located in the position (x,y).
        """
        # we instantiated an array of shape (1,128,128)
        # the same as the number of translations done in our dataset
        resulted_img = np.empty((128,128))
        # for every image in our dataset
        for data in dataset:
            # we get the mean activation for each factor of the latent space
            _, _, mean_batch, _, _ = self.ouput_encoder(data[0])
            # we append the resulting mean of the considered factor in the empty array
            # we created above : 
            resulted_img[data[2].numpy()-1,data[1].numpy()-1] = mean_batch[index_z]
        return resulted_img
        
    def mean_cursor(self, index_z, image):
        """
        index_z : the factor's index of the latent variable we are
                focused on
        image : single image from which we want to see the impact of mean traversal

        Function that returns an array of images reconstructed by modify the learned
        mean of a single image, from deleting -3 to the mean to adding +3. 
        """
        #  we get the informations needed from the encoder
        learned_mean, _, _, learned_var, sample = self.ouput_encoder(image)
        # we initialize the learned mean by the one outward the encoder
        learned_mean_init = learned_mean[index_z]
        # we instanciated an empty array for the resulted images
        generated_img = np.empty((0, 1,128,128,1))
        # for the visualisation, we want to move through latent space
        # by deleting 3 to the learned mean to adding 3
        for i in range(-3,4):
            # we change the learned mean of the considered factor
            # ( we do not want to change the other )
            learned_mean[index_z] = learned_mean_init + i
            # we create a new ditribution corresponding 
            # to the ouput of the encoder
            dist = tfd.MultivariateNormalDiag(
                    loc=learned_mean,
                    scale_diag=learned_var)
            sample = dist.sample((1))
            # we decode the sample of the resulted distribution
            img = self.model.decoder(sample)
            # we get the mean() of the output normal distribution
            # for vizualisation
            img_mean = img.mean()
            # we append the result in the array created above
            generated_img = np.append(generated_img, [img_mean], axis=0)
        return generated_img

    def get_imgs_to_plot(self, sorted_variances, image, 
                        dataset_form, dataset_echelle, 
                        dataset_quaternion, dataset_position):
        """
        sorted_variances : list of factors of the latent space associated with
                            their learned variances sorted in a croissant order
        image : a single image from which we want to visualize the imact of mean traveral
        dataset_form : the dataset of images caracterized by their forms  
        dataset_echelle :  the dataset of images caracterized by their scales
        dataset_quaternion :  the dataset of images caracterized by their quaternions
        dataset_position : the dataset of images caracterized by the position of the center of the 
                            object

        Function that returns every data that we will need in order to plot 
        """
        # we initialize empty array
        pos_img_list = np.empty((0,128,128))
        gen_img_list = np.empty((0, 7,1,128,128,1))
        form_list = np.empty((0, 15))
        scale_list = np.empty((0, 15))
        quaternions_list = np.empty((0, 15))
        # we keep only the first lowest sorted factors corresponding
        # to the smallest variances
        sixth_var = sorted_variances[:6]
        
        # for every factor in the keeped ones
        for i, elem in enumerate(sixth_var):
            # we generate the image through the traversal mean
            gen_img = self.mean_cursor(elem[0], image)
            # we get the resulted image for the translations of the object 
            resulted_img = self.position_function(dataset_position, elem[0])
            # we get the values for the plot of each form
            form = self.plot_qualitative_function(dataset_form, elem[0])
            # we get the values for the plot of each scale
            scale = self.plot_qualitative_function(dataset_echelle, elem[0])
            # we get the values for the plot of each quaternion
            quaternion = self.plot_qualitative_function(dataset_quaternion, elem[0])
            
            # we append the results in the corresponding empy array
            pos_img_list = np.append(pos_img_list, [resulted_img], axis=0)
            gen_img_list = np.append(gen_img_list, [gen_img], axis=0)
            form_list = np.append(form_list, [form], axis=0)
            scale_list = np.append(scale_list, [scale],axis=0)
            quaternions_list = np.append(quaternions_list, [quaternion], axis=0)
            
        return pos_img_list, gen_img_list, form_list, scale_list, quaternions_list, sixth_var
    
    def plot_figure(self, pos_img_list, gen_img_list, form_list, 
                    scale_list, quaternions_list, sixth_var):
        """
        pos_img_list : a list of images that shows the activation of each factor
                    of the latent variable depending on the position of the object on the image
        gen_img_list : a list of images generated by moving the learned mean of each factor
        form_list :  a list of numbers representing the learned mean of each factor for different forms
        scale_list : a list of numbers representing the learned mean of each factor for different scales
        quaternions_list :  a list of numbers representing the learned mean of each factor for different 
                        quaternions
        sixth_var : the six first factors with the smallest variances that we want to visualise

        Function that plots different visualisation allowing us to see the disentanglement of 
        each factor of the latent space.
        """
        fig, axs = plt.subplots(nrows=10, ncols=6, figsize=(10, 10))
        fig.suptitle('Visualisation of latent space', fontsize=16)
        # we plot the label corresponding to the specific line
        axs[0,0].set_ylabel('position')
        axs[1,0].set_ylabel('form')
        axs[2,0].set_ylabel('scale')
        axs[3,0].set_ylabel('quaternion')
        axs[7,0].set_ylabel('-3 --------------------- mean latent traversal ----------------- +3')
        # for every factor contained in sixth_var
        for i, n in enumerate(sixth_var):
            # we plot the image obtained for the translations
            axs[0,i].imshow(pos_img_list[i], cmap='hsv')
            axs[0,i].set_title("factor : " + str(n[0]))
            # we plot the curve obtained for the forms
            axs[1,i].plot(form_list[i])
            axs[1,i].set(ylim=(-2, 2))
            # we plot the curve obtained for the scales
            axs[2,i].plot(scale_list[i])
            axs[2,i].set(ylim=(-2, 2))
            # we plot the curve obtained for the quaternions
            axs[3,i].plot(quaternions_list[i])
            axs[3,i].set(ylim=(-2, 2))
            axs[9,i].set_xlabel("variance : \n" + str(n[1][0]))
            for j in range(6):
                # we plot the images generated by the traversal mean 
                # for each corresponding factor
                axs[j+4,i].imshow(gen_img_list[i,j,0,:,:,0],  cmap='gray')
        for i in range(10):
            for j in range(6):
                # we hide the tick labels for every image
                plt.setp(axs[i,j].get_xticklabels(), visible=False)
                plt.setp(axs[i,j].get_yticklabels(), visible=False)
                axs[i,j].tick_params(axis='both', which='both', length=0)
        plt.show()


    #########################################################################################
            # The function below are used for quantitative 
            # visualiation of latent space
    #########################################################################################

    def plot_curves_quantitative(self, history, epochs):
        """
        history : a keras object that keeps in memory different values 
                of training and validation loss 
        epochs : the number of epochs for training

        Function that plots losses and accuracy for training and 
        testing set
        """
        # we get the training and validation accuracy from
        # the history object
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        # we get the training and validation loss from
        # the history object
        loss=history.history['loss']
        val_loss=history.history['val_loss']

        epochs_range = range(epochs)
        # we plot the curves of accuracies depending on the epoch
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        # we plot the curves of losses depending on the epoch
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    def map_dataset_quantitative(self, filepath):
        """
        filepath : path of the image we need to decode

        Function that returns a decoded image
        """
        img = tf.io.read_file(filepath)
        # we decode the image in int for 1 channel (grayscale)
        img = tf.image.decode_png(img, channels=1, dtype=tf.uint8)
        # we scale each pixel of the image between 0 and 1
        img = tf.image.convert_image_dtype(img, tf.float32)
        # we resize the image to the shape (1,128,128)
        img = tf.image.resize(img, [128,128])
        return img

    def create_dataset_quantitative_visualisation(self):
        """
        Function that returns a list of training and testing sets for each 
        dataset corresponding to our different factors (labels) that we want to
        predict.
        For each dataset, we get a list of filepath corresponding to every image in the dataset, 
        we map each filepath to decode and get the image, we transform the map object to an iterable list. 
        We shuffle the dataset, and we take the first 70% to create the training set and the rest for the testing set.
        """
        positionx_path = [f for f in glob.glob(str(self.path_quanti+'/positionx/*.png'), recursive=True)]
        positionx_path = map(self.map_dataset_quantitative, positionx_path)
        positionx = list(positionx_path)
        random.shuffle(positionx)
        train_positionx, test_positionx = positionx[:int(np.ceil(len(positionx)*0.7))], positionx[int(np.ceil(len(positionx)*0.7)):]

        
        positiony_path = [f for f in glob.glob(str(self.path_quanti+'/positiony/*.png'), recursive=True)]
        positiony_map = map(self.map_dataset_quantitative, positiony_path)
        positiony = list(positiony_map)
        random.shuffle(positiony)
        train_positiony, test_positiony = positiony[:int(np.ceil(len(positiony)*0.7))], positiony[int(np.ceil(len(positiony)*0.7)):]

        
        form_path = [f for f in glob.glob(str(self.path_quanti+'/form/*.png'), recursive=True)]
        form_map = map(self.map_dataset_quantitative, form_path)
        form = list(form_map)
        random.shuffle(form)
        train_form, test_form = form[:int(np.ceil(len(form)*0.7))], form[int(np.ceil(len(form)*0.7)):]

        scale_path = [f for f in glob.glob(str(self.path_quanti+'/scale/*.png'), recursive=True)]
        scale_map = map(self.map_dataset_quantitative, scale_path)
        scale = list(scale_map)
        random.shuffle(scale)
        train_scale, test_scale = scale[:int(np.ceil(len(scale)*0.7))], scale[int(np.ceil(len(scale)*0.7)):]
    

        quaternion_path = [f for f in glob.glob(str(self.path_quanti+'/quaternion/*.png'), recursive=True)]
        quaternion_map = map(self.map_dataset_quantitative, quaternion_path)
        quaternions = list(quaternion_map)
        random.shuffle(quaternions)
        train_quaternion, test_quaternion = quaternions[:int(np.ceil(len(quaternions)*0.7))], quaternions[int(np.ceil(len(quaternions)*0.7)):]

        return train_quaternion, test_quaternion, train_scale, test_scale, train_form, test_form, train_positiony, test_positiony, train_positionx, test_positionx


    def generator(self, nb_batch, positionx, positiony, form, scale, quaternions):
        """
        nb_batch: number of images used to create a avrage latent variable
        positionx: dataset caracterizing the position of the objetc along x-axis
        positiony: dataset caracterizing the position of the objetc along y-axis
        form: dataset caracterizing the form of the object
        scale: dataset caracterizing the scale of the object
        quaternions: dataset caracterizing the quaternion of the object
        """
        # the factors correspond to the label of each dataset
        factors = [0,1,2,3,4]
        # as we use the keras CategoricalCrossentropy() loss during the training, we
        # need to one-hot encode our labels
        labels = tf.keras.utils.to_categorical(factors, num_classes=5)
        # each label is mapped with one dataset
        data_label = [[labels[0],form], [labels[1],scale],[labels[2],quaternions],[labels[3],positiony], 
                [labels[4],positionx]]
        # the generator need to be infinite, it will be stopped 
        # during the model.fit() keras' function
        while True:
            # we sample a single random factor
            data = random.sample(data_label,  1)
            # we get the corresponding label
            label = data[0][0]
            samples = []
            pairs_sample = []
            diff_sample = []
            # for every image in the sample, we get nb_batch of these images
            for img in random.sample(data[0][1],  nb_batch):
                # we endode the image, sample the encoded distribution and append
                # it in the array created above
                latent = self.model.encoder(img, True)
                latent_sample = latent.sample()
                samples.append(latent_sample)
            # for every two steps in nb_batch
            for i in range(1, nb_batch, 2):
                # we get a first element in the samples array
                one = samples[i - 1]
                # we get a second element in the same array
                two = samples[i]
                # we create pairs with these two elements
                pairs_sample.append((one, two))
            # for every pair in the array
            for pair in pairs_sample:
                # we compute the absolute difference
                diff = abs(np.array(pair[0])-np.array(pair[1]))
                # we append this difference in the array created abose
                diff_sample.append(diff)
            # we return the mean of all these differences, corresponding to an
            # "averaged" latent space, and the label reshaped corresponding. 
            yield (np.array(diff_sample).mean(axis=0), label.reshape(-1,1).T)
    