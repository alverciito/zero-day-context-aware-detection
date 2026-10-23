import numpy as np

def create_feature_map(n, clusters):
    feature_map = []
    available_choices = np.arange(n)
    shuffled = np.random.permutation(available_choices)
    for i in range(clusters):
        feature_map.append(shuffled[i::clusters])
    return feature_map
