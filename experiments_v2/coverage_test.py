import numpy as np
import experiments_v2.helper_functions as helper

def _createGaussians(cluster_samples):
    data = []
    center_coords = np.array([[-1,-1],
                     [-1, 1],
                     [1, 1],
                     [1, -1]])
    for i in range(center_coords.shape[0]):
        center_coord = center_coords[i]*10
        cluster_points = np.random.normal(loc=center_coord, scale=0.1, size=[cluster_samples, 2])
        data.extend(cluster_points)

    return np.array(data)

def getData(real_samples, fake_samples):
    real_cluster_samples = real_samples // 4
    real_features = _createGaussians(cluster_samples=real_cluster_samples)
    if real_features is not None:
        fake_features = np.random.normal(loc=0, scale=1, size=[fake_samples, 2])

    return real_features, fake_features

def experimentCoverage(real_features, fake_features, k_val):
    boundaries_real, boundaries_fake, distance_matrix_pairs = helper.getBoundaries(real_features, fake_features, k_val)
    precision, recall, density, coverage = helper.getScores(distance_matrix_pairs, boundaries_fake, boundaries_real, k_val)
    scores = np.array([precision, recall, density, coverage])

    return scores

def getClusterData(cluster_samples):
    data = []
    center_coords = np.array([[-1, -1],
                              [-1, 1],
                              [1, 1],
                              [1, -1]])
    for i in range(center_coords.shape[0]):
        center_coord = center_coords[i] * 1
        cluster_points = np.random.normal(loc=center_coord, scale=0.1, size=[cluster_samples, 2])
        data.append(cluster_points)

    return data




