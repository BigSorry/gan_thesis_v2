import matplotlib.pyplot as plt
import numpy as np
import experiments_v2.helper_functions as helper
import visualize as plotting

def _createClusters(cluster_samples, weight):
    data = []
    xy_min = [0, 0]
    xy_max = [1, 1]
    xy_min2 = [0, 0]
    xy_max2 = [1, -1]
    xy_min3 = [0, 0]
    xy_max3 = [-1, -1]
    xy_min4 = [0, 0]
    xy_max4 = [-1, 1]
    clusters = np.array([[xy_min, xy_max], [xy_min2, xy_max2],
                         [xy_min3, xy_max3], [xy_min4, xy_max4]])

    for i in range(len(clusters)):
        begin_params = clusters[i][0]*weight
        end_params = clusters[i][1]*weight
        cluster_points = np.random.uniform(low=begin_params, high=end_params, size=(cluster_samples, 2))
        data.extend(cluster_points)

    return np.array(data)

def getDataClusters(real_samples, fake_samples, weight):
    real_cluster_samples = real_samples // 4
    fake_cluster_samples = fake_samples // 4
    real_features = _createClusters(cluster_samples=real_cluster_samples, weight=1)
    fake_features = _createClusters(cluster_samples=fake_cluster_samples, weight=weight)

    return real_features, fake_features

def getData(start, high, real_samples, fake_samples, adjust):
    # Now act as gaussian clusters

    real_features = np.random.uniform(low=start, high=high, size=(real_samples, 2))
    fake_features = np.random.uniform(low=start+adjust, high=high-adjust, size=(fake_samples, 2))

    return real_features, fake_features

def getScores(real_features, fake_features, k_val):
    boundaries_real, boundaries_fake, distance_matrix_pairs = helper.getBoundaries(real_features, fake_features, k_val)
    precision, recall, density, coverage = helper.getScores(distance_matrix_pairs, boundaries_fake, boundaries_real, k_val)
    scores = np.array([precision, recall, density, coverage])

    return scores, np.mean(boundaries_real), np.mean(boundaries_fake)

def plotResults(scores, radi, subplots):
    fig, ax = plt.subplots(subplots)
    scores_np = np.array(scores)
    recalls = scores_np[:, 1]
    plotting.plotErrorbar(ax[0], "recall", radi, recalls)
    coverages = scores_np[:, 3]
    plotting.plotErrorbar(ax[1], "coverage", radi, coverages)

    if subplots > 2:
        precisions = scores_np[:, 0]
        plotting.plotErrorbar(ax[2], "precision", radi, precisions)
        densities = scores_np[:, 2]
        plotting.plotErrorbar(ax[3], "density", radi, densities)


