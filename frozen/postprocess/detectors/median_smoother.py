import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

def median_smoothing(detector, window_length):
    """
    Applies median smoothing to the detector and target signals.

    :param detector: np.array, detection signal
    :param window_length: int, window length
    :return: (np.array, np.array) smoothed detection
        -   smoothed detection signal
    """
    detection_smooth = median_filter(detector, size=2 * window_length + 1, mode='nearest')
    return detection_smooth

def mean_smoothing(detector, window_length):
    """
    Aplica un suavizado de media a las señales de detector y target.

    :param detector: np.array, señal de detección
    :param window_length: int, longitud de la ventana para el suavizado
    :return: (np.array, np.array) señales suavizadas de detección y objetivo
    """
    detection_smooth = uniform_filter1d(detector, size=2 * window_length + 1, mode='nearest')
    return detection_smooth

def label_smoother(labels, window_length):
    """
    Applies median smoothing to the labels.

    :param labels: np.array, labels
    :param window_length: int, window length
    :return: np.array smoothed labels
    """
    smooth = 0
    cnt = 0
    new_labels = list()

    for idx, label in enumerate(labels):
        if label == 1:
            smooth = 1
            cnt = 0
        else:
            cnt += 1
            if cnt > window_length:
                smooth = 0
        new_labels.append(smooth)
    if sum(new_labels) <= 0:
        raise RuntimeError("No labels after smoothing.")
    return np.array(new_labels)
