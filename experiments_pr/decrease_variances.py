import numpy as np
import metrics.metrics as mtr
import matplotlib.pyplot as plt

def getData(sample_count, mean_vec, fake_covariance):
    dimension = mean_vec.shape[0]
    real_covariance = np.eye(dimension)
    real_features = np.random.multivariate_normal(mean_vec, real_covariance, size=sample_count)
    fake_features = np.random.multivariate_normal(mean_vec, fake_covariance, size=sample_count)

    return real_features, fake_features

def getParams(dimension, ratio):
    mean_vec = np.zeros(dimension)
    cov_matrix = np.eye(dimension)
    zeros = int(ratio * dimension)
    zero_variances = np.random.choice(dimension, zeros, replace=False)
    cov_matrix[zero_variances, zero_variances] = 0

    return mean_vec, cov_matrix

def subspaceExperimentOne(sample_count, dimension, ratio,
                          method_name, params):
        mean_vec, cov_matrix = getParams(dimension, ratio)
        #print(ratio, np.diag(cov_matrix))
        real_features, fake_features = getData(sample_count, mean_vec, cov_matrix)
        pr_score, curve, cluster_labels = mtr.getPRCurve(real_features, fake_features, method_name, params)

        #plotData(real_features, fake_features)

        return pr_score, curve, cluster_labels


def plotData(real_features, fake_features):
    plt.figure()
    plt.scatter(real_features[:, 0], real_features[:, 1])
    plt.figure()
    plt.scatter(fake_features[:, 0], fake_features[:, 1])
    plt.show()
