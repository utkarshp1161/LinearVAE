import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Flatten, Reshape, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from src.model import CustomVAE
from src.utils import plot_latent_space, plot_latent_embeddings_with_labels


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1]) / 255.
    x_test = np.reshape(x_test, [-1, image_size, image_size, 1]) / 255.


    # Convert the NumPy arrays to TensorFlow Dataset objects
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # Batch the data
    batch_size = 128
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    test_dataset = test_dataset.batch(batch_size, drop_remainder=True)

    latent_dim = 2  # Dimensionality of the latent space

    inputs = Input(shape=(image_size, image_size, 1))
    x = Conv2D(32, 3, activation='relu', strides=2, padding='same')(inputs)
    x = Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    def sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling)([z_mean, z_log_var])

    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(7 * 7 * 64, activation='relu')(latent_inputs)
    x = Reshape((7, 7, 64))(x)
    x = Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
    x = Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
    outputs = Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)


    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    decoder = Model(latent_inputs, outputs, name='decoder')




    vae = CustomVAE(encoder, decoder, image_size)


    vae.compile(optimizer='adam')
    # vae.fit(x_train, y_train, epochs=30, batch_size=128, validation_data=(x_test, y_test))
    vae.fit(train_dataset, epochs=30, validation_data=test_dataset)

    plot_latent_space(vae,image_size, n=5, figsize= 5)
    plt.savefig('out.png')
    plot_latent_embeddings_with_labels(vae, x_test, y_test, figsize=15)
    plt.savefig('out2.png')
    
    

if __name__ == "__main__":
    main()
