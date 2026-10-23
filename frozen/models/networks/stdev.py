import numpy as np


class StatisticalAnomaly:
    def __init__(self, n_visible, **kwargs):
        """
        This class implements a simple anomaly detector based on the standard deviation of the data.
        :param n_visible: The number of features.
        :param kwargs: Other parameters.
        """
        self.mean = np.zeros(n_visible)
        self.std = np.zeros(n_visible)
        self.norm_max = np.zeros(n_visible)
        self.norm_min = np.zeros(n_visible)


    def train(self, input_x):
        """
        Train the anomaly detection.
        :param input_x: The data as (n_aes, n_features).
        :return: The error deviation.
        """
        x = self.normalize(input_x, is_training=True)
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        return self.execute(input_x)


    def execute(self, x):
        """
        Execute the anomaly detection.
        :param x: The data as (n_aes, n_features).
        :return: The error deviation.
        """
        x_norm = self.normalize(x)
        error = np.abs(x_norm - self.mean) / (self.std + 1e-16)
        sum_error = np.sum(error, axis=1)
        return sum_error


    def normalize(self, x: np.ndarray, is_training=False):
        """
        Normalize the data.
        :param x: The data as (n_aes, n_features).
        :param is_training: If True, the normalization is done in training mode.
        :return:
        """
        if is_training:
            self.norm_max = np.max(x, axis=0)
            self.norm_min = np.min(x, axis=0)
        norm = (x - self.norm_min) / (self.norm_max - self.norm_min + 1e-16)
        return norm
