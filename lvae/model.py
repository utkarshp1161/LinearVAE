import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Dense, Lambda, Flatten, Reshape, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from keras import backend as K
from keras.losses import binary_crossentropy



class CustomVAE(Model):
    """
       Linear Variational Autoencoder (VAE) class.

       Args:
       - encoder: Encoder model.
       - decoder: Decoder model.
       - size: Size of the input data.
       - **kwargs: Additional keyword arguments.

       Methods:
       - call(inputs): Forward pass of the VAE.
       - train_step(data): Custom training step for the VAE.
       - plot_latent_space(image_size, n=30, figsize=15): Plot the latent space.

       Attributes:
       - encoder: Instance of the encoder model.
       - decoder: Instance of the decoder model.
       - size: Size of the input images.
       """

    def __init__(self, encoder, decoder, size, **kwargs):
        super(CustomVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.size = size

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed

    def train_step(self, data):
        if isinstance(data, tuple):
            x, y = data
        else:
            x = data
            y = None

        # Check if labels are provided
        if y is None:
            raise ValueError("Digit labels are required for custom training.")

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstructed = self.decoder(z)

            # Reconstruction loss
            reconstruction_loss = binary_crossentropy(K.flatten(x), K.flatten(reconstructed))
            reconstruction_loss *= self.size

            # KL divergence
            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            kl_loss = -0.5 * K.sum(kl_loss, axis=-1)

            # Custom loss
            batch_size = tf.shape(x)[0]
            y_true = tf.cast(y[:batch_size], dtype='float32')
            custom_loss = K.mean(K.square(z[:, 0] - y_true), axis=-1) * 10

            # Total loss
            total_loss = K.mean(reconstruction_loss + kl_loss + custom_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {'loss': total_loss, 'reconstruction_loss': reconstruction_loss, 'kl_loss': kl_loss,
                'custom_loss': custom_loss}
