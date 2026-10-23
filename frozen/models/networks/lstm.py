from ..__special__ import logging, HASTE_VALUE, EXECUTION_BS, BATCH_SIZE
from ..utils import create_windowed_data, create_windowed_data_ar
import tensorflow as tf
import numpy as np


class LSTMAutoencoder:
    def __init__(self, n_visible=5, n_hidden=3, lr=0.001, corruption_level=0.0, hidden_ratio=None, sequence_length=500,
                 seed=1234, ar=False, haste: bool = True, name="Anonymous dA"):
        """
        Denoising Autoencoder (dA) class.
        :param n_visible: number of units in visible (input) layer
        :param lr: learning rate
        :param corruption_level: drop-out probability
        :param hidden_ratio: ratio of hidden units to visible units (if None, n_hidden is used)
        :param n_hidden: number of units in hidden layer
        :param seed: random seed
        :param sequence_length: The length of the input window.
        :param ar: If True, the model will be an autoregressive model.
        :param haste: If True, the model will be trained in a single epoch.
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

        self.create_windowed = create_windowed_data_ar if ar else create_windowed_data
        self.haste = 1 if haste else HASTE_VALUE

        self.window = np.zeros((sequence_length, n_visible))
        self.sequence_length = sequence_length

        sequence_input = tf.keras.Input(shape=(sequence_length, n_visible))
        lstm_out = tf.keras.layers.LSTM(self.n_hidden, return_sequences=False)(sequence_input)
        self.encoder = tf.keras.Model(inputs=sequence_input, outputs=lstm_out, name='encoder')

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(self.n_hidden, activation='relu'),
            tf.keras.layers.Dense(n_visible, activation='sigmoid')
        ], name='decoder')

        self.model = tf.keras.models.Sequential([self.encoder, self.decoder], name='LSTMAutoencoder')

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        loss = tf.keras.losses.MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss=loss)
        self.model.build(input_shape=(None, sequence_length, n_visible))

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
        sequential_x, self.window = self.create_windowed(x, self.sequence_length, self.window, batch_size=BATCH_SIZE)
        self.model.fit(sequential_x, epochs=self.haste, verbose=1)
        logging.info(f"Training {self.name}... Done!")
        sequential_x, self.window = self.create_windowed(x, self.sequence_length, self.window, batch_size=EXECUTION_BS)
        mse_per_sample = list()
        for (x_batch, y_batch) in sequential_x:
            predicted = self.model.predict(x_batch, verbose=0)
            mse_batched = tf.sqrt(tf.reduce_mean(tf.square(y_batch - predicted), axis=1))
            mse_per_sample.extend(mse_batched.numpy())
        return mse_per_sample

    def execute(self, x):
        x = np.clip(x, self.norm_min, self.norm_max)
        x = (x - self.norm_min) / (self.norm_max - self.norm_min + 1e-16)
        logging.info(f"Executing {self.name} with {len(x)} instances...")
        sequential_x, self.window = self.create_windowed(x, self.sequence_length, self.window, batch_size=EXECUTION_BS)

        mse_per_sample = list()
        for (x_batch, y_batch) in sequential_x:
            predicted = self.model.predict(x_batch, verbose=0)
            mse_batched = tf.sqrt(tf.reduce_mean(tf.square(y_batch - predicted), axis=1))
            mse_per_sample.extend(mse_batched.numpy())
        return np.array(mse_per_sample)
