import numpy as np
from sklearn.cluster import KMeans as skKMeans

class KMeans:
    def __init__(self, n):
        self.n = n
        self.buffer = list()

    def update(self,x):
        self.buffer.append(x)

    def cluster(self, n_clusters):
        x = np.array(self.buffer)
        self.buffer = list()
        kmeans = skKMeans(n_clusters=n_clusters, random_state=0).fit(x.T)
        sk_labels = kmeans.labels_

        feature_map = list()
        for i in range(n_clusters):
            feature_map.append(np.where(sk_labels == i)[0].tolist())
        return feature_map
