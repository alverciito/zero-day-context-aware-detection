import numpy as np
from ..__special__ import logging
from sklearn.cluster import DBSCAN as skDBSCAN


class DBSCAN:
    def __init__(self, n):
        self.n = n
        self.buffer = list()

    def update(self,x):
        self.buffer.append(x)

    def cluster(self, n_clusters):
        x = np.array(self.buffer)
        x = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0) + 1e-16)
        self.buffer = list()

        current_clusters = -1
        eps = 1.4
        sk_labels = None
        momentum = 1
        __epoch = 0
        while current_clusters != n_clusters and __epoch < 100:
            kmeans = skDBSCAN(eps=eps, min_samples=8).fit(x.T)
            sk_labels = kmeans.labels_
            current_clusters = len(set(sk_labels))
            diff = n_clusters - current_clusters
            eps += 0.1 * diff * momentum
            momentum *= 0.99
            if eps < 0:
                eps = 0.05
            __epoch += 1
            if __epoch == 100:
                logging.warning(f'DBSCAN: Could not find the desired number of clusters: {n_clusters}. '
                                f'Returning the best result: {current_clusters}.')

        sk_labels = np.array(sk_labels) + 1

        feature_map = list()
        for i in range(n_clusters):
            feature_map.append(np.where(sk_labels == i)[0].tolist())

        cluster_lens = [len(cluster) for cluster in feature_map]
        while min(cluster_lens) < 8:
            idx_max_cluster = np.argmax(cluster_lens)
            idx_min_cluster = np.argmin(cluster_lens)
            for _ in range(8):
                feature_map[idx_min_cluster].append(feature_map[idx_max_cluster].pop(-1))
            cluster_lens = [len(cluster) for cluster in feature_map]

        return feature_map
