from keras import backend as K, losses
from keras.layers import Dense, Reshape, Input, Lambda
from keras.models import Model

from MNIST.setup_mnist import MNIST


def create_vae_model(original_dim=784, intermediate_dim=64, latent_dim=32):
    inputs = Input(shape=(28, 28, 1))
    reshaped_inputs = Reshape((original_dim,))(inputs)
    h = Dense(intermediate_dim, activation='relu')(reshaped_inputs)
    z_mean = Dense(latent_dim)(h)
    z_log_sigma = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                  mean=0., stddev=0.1)
        return z_mean + K.exp(z_log_sigma) * epsilon

    z = Lambda(sampling)([z_mean, z_log_sigma])
    encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(original_dim, activation='sigmoid')(x)
    decoder = Model(latent_inputs, outputs, name='decoder')

    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')

    reconstruction_loss = losses.binary_crossentropy(reshaped_inputs, outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    mnist = MNIST()
    x_train = mnist.train_data
    x_valid = mnist.validation_data

    vae.fit(x_train, x_train, epochs=100, batch_size=32, validation_data=(x_valid, x_valid))
    encoder.save_weights('encoder_weights.h5')


def mnist_encoder(original_dim=784, intermediate_dim=64, latent_dim=32, weights=None):
    inputs = Input(shape=(28, 28, 1))
    reshaped_inputs = Reshape((original_dim,))(inputs)
    h = Dense(intermediate_dim, activation='relu')(reshaped_inputs)
    z_mean = Dense(latent_dim)(h)
    z_log_sigma = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                  mean=0., stddev=0.1)
        return z_mean + K.exp(z_log_sigma) * epsilon

    z = Lambda(sampling)([z_mean, z_log_sigma])
    encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

    encoder.load_weights(weights)
    return encoder
