# -*- coding: utf-8 -*-

####
# JOUFFROY Emma stagiaire 2020
# https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/vae.py
# https://www.tensorflow.org/tutorials/generative/cvae
# https://github.com/lukaszbinden/spatial-broadcast-decoder
####

import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
tfd = tfp.distributions

class VariationalAutoEncoder(tf.keras.Model):
  """
  This class creates the encoder and the decoder of the 
  architecture of a variational auto encoder.
  This class is a subclassing of "tf.Keras.Model" in order to 
  write a custom keras model. We will train the outer object of this 
  class, which have a built-in training, evaluation and prediction loops
  ( that we won't use), have all layers properties and from which we can save 
  weights. 
  """
  def __init__(self, latent_size, beta, output_shape_img, name='autoencoder', **kwargs):
    super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
    """
    latent_size : size of the latent space 
    beta: value of beta for the loss
    output_shape_img: shape of the image 
    generative_net : sequential network of the encoder
    inference_net : sequential networf of the decoder
    """
    self.latent_size = latent_size
    self.beta = beta
    self.output_shape_img=output_shape_img 

    self.generative_net = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            filters=12, kernel_size=3, strides=1, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=1, activation='relu',padding='same'),
        tf.keras.layers.Conv2D(
            self.output_shape_img[-1],kernel_size=1, strides=1, activation=None,padding='same'),
    ])
    self.inference_net = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=2, activation='relu',padding='same'),
        tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=2, activation='relu',padding='same'),
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=2, activation='relu',padding='same'),
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=2, activation='relu',padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2*self.latent_size),
    ])

  @tf.function
  def _softplus_inverse(self, x):
    """
    helper which computes the function inverse of `tf.nn.softplus` 
    """
    return tf.math.log(tf.math.expm1(x))

  def sample(self, batch):
    """
    helper which computes the encoding and the decoding of a batch of 
    images 

    batch : batch of images to pass through the model, without learning
    """
    latent = self.encoder(eps)
    latent_sample = latent.sample()
    return self.decoder(latent_sample)

  def encoder(self, img_batch, add_dims=False):
    """
    function which computes the inference network on a beatch of images
    and returns a density probability of a multivariate normal law which
    parameters are the output of the encoding
    Returns p(z|x)

    img_batch : batch of images pass through the encoder
    add_dims : boolean needed when we encode a signle image 
               (for the input to be rank : 4)
    """
    # in order to compute the encoder on a single image
    # we need to add a dimension such as the input tensor
    # with shape (128, 128, 1) becomes (1, 128, 128, 1). 
    if add_dims == True:
      if(tf.rank(img_batch) != 4):
        img_batch = tf.expand_dims(img_batch, axis=0)
    # we pass the images to the keras sequential layer
    net = self.inference_net(img_batch)
    # returns the multivariate normal distribution of parameters: 
    # loc (mean) which is half of the latent space
    # scale_diag (stddv) which is the softplus of the other half of the 
    # latent space plus a constante
    self.z_mean = net[..., :self.latent_size]
    self.z_var = tf.nn.softplus(net[..., self.latent_size:] +
                                self._softplus_inverse(1.0))
    return tfd.MultivariateNormalDiag(
        loc=self.z_mean,
        scale_diag=self.z_var, name="code")

  def spatial_broadcasting(self, inputs):
    """
    computes the spatial broadcasting of inputs
    in order to add spatial informations for the decoder

    input : samples of the encoder that we want 
    to decode
    """
    # getting the batch size
    batch_size = tf.shape(inputs)[0]
    # getting the feature size
    feature_size = tf.shape(inputs)[1]
    # getting height and width of the image for the tiling (=128)
    d = w = self.output_shape_img[0]
    # tiling inputs 
    z_b = tf.tile(inputs, [1, d * w])
    # reshaping inputs to get the size of the original image
    z_b = tf.reshape(z_b, [batch_size, d, w, feature_size])
    # creating linspace of constants of size (1,1,128)
    x = tf.linspace(tf.constant(-1, tf.float32), tf.constant(1, tf.float32), w)
    y = tf.linspace(tf.constant(-1, tf.float32), tf.constant(1, tf.float32), w)
    # creating meshgrid of x and y in order to concatenate them
    xb, yb = tf.meshgrid(x, y)
    # expanding dims in order to get the same shape as z_b
    xb = tf.expand_dims(xb, 2)
    yb = tf.expand_dims(yb, 2)
    # function that concatenates e with xb and yb
    def pe(e):
      res = tf.concat(axis=2, values=[e, xb, yb])
      return res
    # using a map function to apply the pe function on z_b
    z_sb = tf.map_fn(lambda m: pe(m), z_b)
    return z_sb

  def decoder(self, codes):
    """
    Decodes latent features sample as a multinomial distribution
    Returns p(x|z)

    codes: sample from the encoder's output that we want to decode
    """
    # we apply the spatial broadcasting to the input
    codes =  self.spatial_broadcasting(codes)
    # we apply the generative_net to the ouput of the spatial broadcasting 
    logits = self.generative_net(codes)
    # return an independant normal distribution for each pixel of each images
    # with parameters : mean = logits and scale = 0.2
    return tfd.Independent(
            distribution=tfd.Normal(loc=logits, scale=0.02),
            reinterpreted_batch_ndims=3, 
            name="image")

  def make_mixture_prior(self):
    """
    returns a probability distribution of a multivariate 
    normal law for the prior
    Returns p(x)
    """
    return tfd.MultivariateNormalDiag(
        loc=tf.zeros([self.latent_size]),
        scale_identity_multiplier=1.0)







