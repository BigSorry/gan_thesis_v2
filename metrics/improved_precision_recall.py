import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

def getPrecisionRecall(real_features, fake_features, k):
    distance_matrix_real = pairwise_distances(real_features, real_features, metric='euclidean')
    distance_matrix_fake = pairwise_distances(fake_features, fake_features, metric='euclidean')
    distance_matrix_pairs = pairwise_distances(real_features, fake_features, metric='euclidean')
    distance_matrix_real = np.sort(distance_matrix_real, axis=1)
    distance_matrix_fake = np.sort(distance_matrix_fake, axis=1)
    boundaries_real = distance_matrix_real[:, k]
    boundaries_fake = distance_matrix_fake[:, k]

    # Code reference
    # https://github.com/clovaai/generative-evaluation-prdc/blob/master/prdc/prdc.py
    precision = (distance_matrix_pairs < np.expand_dims(boundaries_real, axis=1)).any(axis=0).mean()
    recall = (distance_matrix_pairs < np.expand_dims(boundaries_fake, axis=0)).any(axis=1).mean()

    return np.array([precision, recall])

