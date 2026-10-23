import tensorflow as tf
import numpy as np
from ..__special__ import EXECUTION_BS

def create_windowed_data(x: np.ndarray, window_size: int, padding_window: np.ndarray, batch_size: int = 1):
    """
    Create a context-aware windowed dataset from a numpy array.
    :param x: The numpy array.
    :param window_size: The size of the window.
    :param padding_window: The previous last window.
    :param batch_size: The batch size.
    :return: The windowed dataset and the last window.
    """
    full_x = np.concatenate([padding_window, x], axis=0)
    dataset = tf.data.Dataset.from_tensor_slices(full_x[1:])
    dataset = dataset.window(size=window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    dataset = dataset.map(lambda window: (window, window[-1]))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    last_window = full_x[-window_size:]
    dataset.specs = (tf.TensorSpec(shape=(None, window_size, x.shape[-1]), dtype=tf.float32),
                     tf.TensorSpec(shape=(None, x.shape[-1]), dtype=tf.float32))
    return dataset, last_window

def create_windowed_data_ar(x: np.ndarray, window_size: int, padding_window: np.ndarray, batch_size: int = 1):
    """
    Create an autoregressive windowed dataset from a numpy array.
    :param x: The numpy array.
    :param window_size: The size of the window.
    :param padding_window: The previous last window.
    :param batch_size: The batch size.
    :return: The windowed dataset and the last window.
    """
    window_size += 1
    full_x = np.concatenate([padding_window, x], axis=0)
    dataset = tf.data.Dataset.from_tensor_slices(full_x)
    dataset = dataset.window(size=window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    last_window = full_x[-(window_size - 1):]
    dataset.specs = (tf.TensorSpec(shape=(None, window_size, x.shape[-1]), dtype=tf.float32),
                     tf.TensorSpec(shape=(None, x.shape[-1]), dtype=tf.float32))
    return dataset, last_window

def create_dataset(x: np.ndarray, window_size: int = None, padding_window: np.ndarray = None, batch_size: int = 1):
    """
    Create a dataset from a numpy array.
    :param x: The numpy array.
    :param window_size: The size of the window. (NOT USED - LEGACY)
    :param padding_window: The previous last window. (NOT USED - LEGACY)
    :param batch_size: The batch size.
    """
    dataset = tf.data.Dataset.from_tensor_slices(x)
    dataset = dataset.map(lambda in_x: (in_x, in_x))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset, None



if __name__ == '__main__':
    x_test = np.tile(np.arange(1000), (6, 1)).T
    windowed_data, lw = create_windowed_data_ar(x_test, 10, -np.ones((10, 6)), batch_size=EXECUTION_BS)
    total_len = 0
    for window_x, y in windowed_data:
        total_len += len(window_x)
        print(y)

    windowed_data_2, lw_2 = create_windowed_data(x_test, 10, lw, batch_size=EXECUTION_BS)
    for window_x, y in windowed_data:
        total_len += len(window_x)
        print(y)
    print(total_len)
