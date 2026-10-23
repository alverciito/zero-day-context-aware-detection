import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def stat_detector(frame: np.ndarray, window_length: int = 10_000, ref_period: tuple = (100_000, 200_000)):
    """
    Detect statistics deviations in a given frame using an efficient rolling window approach.

    :param frame: np.array, input signal.
    :param window_length: int, window size.
    :param ref_period: tuple, reference period (start, end).
    :return: np.array, deviation scores.
    """
    ref_mean = np.mean(frame[ref_period[0]:ref_period[1]])
    ref_std = np.std(frame[ref_period[0]:ref_period[1]])

    kernel = np.ones(window_length) / window_length
    mean_values = np.convolve(frame, kernel, mode='same')

    squared_frame = frame ** 2
    mean_squared_values = np.convolve(squared_frame, kernel, mode='same')

    std_values = np.sqrt(mean_squared_values - mean_values**2)

    mean_dev = (mean_values - ref_mean) ** 2
    std_dev = (std_values - ref_std) ** 2
    detection_score = 2 * mean_dev * std_dev / (mean_dev + std_dev + 1e-9)

    retval = np.array(detection_score)
    retval[:ref_period[-1]] = 0
    retval[-window_length:] = 0
    retval = np.roll(retval, -window_length // 2)
    return retval


def kmeans_detector(frame: np.ndarray, window_length: int = 10_000, ref_period: tuple = (100_000, 200_000), k: int = 10):
    """

    :param frame:
    :param window_length:
    :param ref_period:
    :param k:
    :return:
    """
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=0)
    kmeans.fit(frame[ref_period[0]:ref_period[1]].reshape(-1, 1))
    distances = kmeans.transform(frame.reshape(-1, 1))
    min_distances = np.min(distances, axis=1)
    return min_distances


def gmm_detector(frame: np.ndarray, window_length: int = 10_000, ref_period: tuple = (100_000, 200_000), k: int = 10):
    """
    Fits a Gaussian Mixture Model to a reference period within the data and calculates the minimum Mahalanobis distance
    from each data point in the frame to the nearest Gaussian component's center.

    :param frame: The full dataset as a 1D numpy array.
    :param window_length: The length of the window for analyzing anomalies (not used in this function but included for consistency).
    :param ref_period: A tuple indicating the start and end indices of the reference period in the data.
    :param k: The number of components (Gaussian distributions) to fit.
    :return: An array of minimum Mahalanobis distances from each point to the nearest Gaussian component.
    """
    gmm = GaussianMixture(n_components=k, covariance_type='full', n_init=10, max_iter=300, random_state=0)
    gmm.fit(frame[ref_period[0]:ref_period[1]].reshape(-1, 1))

    distances = []
    for i in range(len(frame)):
        data_point = frame[i].reshape(-1, 1)
        min_dist = np.inf
        for j in range(k):
            mean = gmm.means_[j]
            cov_inv = np.linalg.inv(gmm.covariances_[j])
            diff = data_point - mean
            mahalanobis_dist = np.sqrt(np.dot(np.dot(diff.T, cov_inv), diff))
            if mahalanobis_dist < min_dist:
                min_dist = mahalanobis_dist
        distances.append(min_dist[0, 0])

    return np.array(distances)
