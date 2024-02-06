import numpy as np

from tensorflow.keras import backend as K
import tensorflow.keras as keras
from keras import layers


def set_1D_encoder(input_shape, latent_dim=2):
  encoder_inputs = keras.Input(shape=(input_shape,))
  x = layers.Dense(128, activation="relu")(encoder_inputs)
  z_mean = layers.Dense(latent_dim, name="z_mean")(x)
  z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

  def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

  z = layers.Lambda(sampling)([z_mean, z_log_var])
  encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
  return encoder

def set_1Ddecoder(output_shape, latent_dim=2):
  latent_inputs = keras.Input(shape=(latent_dim,))
  x = layers.Dense(128, activation="relu")(latent_inputs)
  decoder_outputs = layers.Dense(output_shape, activation='sigmoid')(x)
  decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
  return decoder