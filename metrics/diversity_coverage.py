import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

# Rows are real samples and columns are generated samples
def getDensityRecall(real_features, fake_features, k):
    distance_matrix_real = pairwise_distances(real_features, real_features, metric='euclidean')
    distance_matrix_real = np.sort(distance_matrix_real, axis=1)
    distance_matrix_pairs = pairwise_distances(real_features, fake_features, metric='euclidean')
    boundaries_real = distance_matrix_real[:, k]

    # Code reference
    # https://github.com/clovaai/generative-evaluation-prdc/blob/master/prdc/prdc.py
    density = (1. / float(k)) * (distance_matrix_pairs < np.expand_dims(boundaries_real, axis=1)).sum(axis=0).mean()
    coverage = (distance_matrix_pairs.min(axis=1) < boundaries_real).mean()

    return np.array([density, coverage])