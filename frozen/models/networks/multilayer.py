from ..__special__ import logging, HASTE_VALUE, BATCH_SIZE
import tensorflow as tf
import numpy as np


class MLPAutoencoder:
    def __init__(self, n_visible=5, n_hidden=3, lr=0.001, corruption_level=0.0, hidden_ratio=None,
                 seed=1234, haste: bool = True, name="Anonymous dA", **kwargs):
        """
        Denoising Autoencoder (dA) class.
        :param n_visible: number of units in visible (input) layer
        :param lr: learning rate
        :param corruption_level: drop-out probability
        :param hidden_ratio: ratio of hidden units to visible units (if None, n_hidden is used)
        :param n_hidden: number of units in hidden layer
        :param seed: random seed
        :param haste: Use haste mode or not.
        :param name: name of the dA
        """
        self.input_dim = n_visible
        self.lr = lr
        self.dropout = corruption_level
        self.n_hidden = int(np.ceil(hidden_ratio * n_visible) if hidden_ratio is not None else n_hidden)

        self.seed = seed
        self.name = name

        self.norm_min = 0
        self.norm_max = 1

        self.haste = 1 if haste else HASTE_VALUE


        self.encoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.n_hidden, activation='relu', input_shape=(self.input_dim,)),
            tf.keras.layers.Dropout(self.dropout)
        ])
        self.decoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.input_dim, activation='sigmoid')
        ])
        self.model = tf.keras.models.Sequential([self.encoder, self.decoder])

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        loss = tf.keras.losses.MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss=loss)

    def forward_encode(self, x):
        """
        Encodes the input data.
        :param x: The input data
        :return: The encoded data
        """
        return self.encoder(x)

    def forward_decode(self, hidden):
        """
        Decodes the hidden data.
        :param hidden: The hidden data.
        :return: The decoded data.
        """
        return self.decoder(hidden)

    def train(self, input_x):
        """
        Trains the dA model.
        :param input_x: The input data
        :return: The RMSE reconstruction error during training.
        """
        logging.info(f"Training {self.name}...")
        self.norm_min = np.min(input_x, axis=0)
        self.norm_max = np.max(input_x, axis=0)
        x = (input_x - self.norm_min) / (self.norm_max - self.norm_min + 1e-16)
        self.model.fit(x, x, epochs=self.haste, batch_size=BATCH_SIZE, verbose=1)
        z = self.reconstruct(x)
        mse = np.mean((x - z) ** 2, axis=1)
        return mse


    def reconstruct(self, x):
        y = self.forward_encode(x)
        z = self.forward_decode(y)
        return z

    def execute(self, x):
        x = np.clip(x, self.norm_min, self.norm_max)
        x = (x - self.norm_min) / (self.norm_max - self.norm_min + 1e-16)
        z = self.reconstruct(x)
        mse = np.mean((x - z) ** 2, axis=1)
        return mse
