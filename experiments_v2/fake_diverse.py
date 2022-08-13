import matplotlib.pyplot as plt
import numpy as np
import experiments_v2.helper_functions as helper
import visualize as plotting

def getData(real_samples, fake_samples, constant_radius):
    # Now act as gaussian clusters
    clusters = 2
    fake_centers = helper.generateCircles(clusters, radius=constant_radius, dimension=2)
    cluster_samples = fake_samples // clusters
    fake_features = np.zeros((fake_samples, fake_centers.shape[0]))
    for i in range(fake_centers.shape[0]):
        center_vec = fake_centers[i, :]
        cluster_data = np.random.normal(loc=center_vec, scale=0.1, size=[cluster_samples, 2])
        begin_index = i*cluster_samples
        fake_features[begin_index:begin_index+cluster_samples, :] = cluster_data

    if fake_features is not None:
        real_features = np.random.normal(loc=0, scale=1, size=[real_samples, 2])

    return real_features, fake_features

def getDataNew(real_samples, fake_samples, variance, dimension):
    mean = np.zeros(dimension)
    cov = np.eye(dimension)
    real_features = np.random.multivariate_normal(mean, cov, size=real_samples)
    cov*=variance
    fake_features = np.random.multivariate_normal(mean, cov, size=fake_samples)

    return real_features, fake_features

def constantRadii(real_features, fake_features, k_val):
    boundaries_real, boundaries_fake, distance_matrix_pairs = helper.getBoundaries(real_features, fake_features, k_val)
    precision, recall, density, coverage = helper.getScores(distance_matrix_pairs, boundaries_fake, boundaries_real, k_val)
    scores = np.array([precision, recall, density, coverage])

    return scores

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


