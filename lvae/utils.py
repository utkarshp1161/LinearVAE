import matplotlib.pyplot as plt
import numpy as np

def plot_latent_space(vae, image_size, n=30, figsize=15):
    # Display a grid of n*n digits (default 30*30)
    figure = np.zeros((image_size * n, image_size * n))
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(image_size, image_size)
            figure[i * image_size: (i + 1) * image_size,
                   j * image_size: (j + 1) * image_size] = digit

    plt.figure(figsize=(figsize, figsize))
    plt.imshow(figure, cmap='Greys_r')
    plt.axis('off')
    plt.show()



def plot_latent_embeddings_with_labels(vae, data, labels, n=30, figsize=15):
    """Plots n*n MNIST digits as a function of their latent vector along with their class labels.
    """
    encoder = vae.encoder
    z_mean, _, _ = encoder.predict(data)
    
    plt.figure(figsize=(figsize, figsize))

    # Create a scatter plot of the latent variables colored by their class
    for i in range(10):  # Assuming 10 classes for MNIST
        indices = np.where(labels == i)[0]
        indices = np.random.choice(indices, size=n, replace=False)
        plt.scatter(z_mean[indices, 0], z_mean[indices, 1], label=str(i), alpha=0.7)

    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.title('Latent Space Representations with Class Labels')
    plt.legend()
    plt.grid(True)
    plt.show()


