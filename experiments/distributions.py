import numpy as np
from scipy.stats import multivariate_normal
import helper_functions as helper

def getGaussian(sample_size, dimension, other_scale):
    mean_vec = np.zeros(dimension)
    identity_cov = np.eye(dimension)
    reference_distribution = np.random.multivariate_normal(mean_vec, identity_cov, sample_size)
    cov_mat = identity_cov*other_scale
    scaled_distributions = np.random.multivariate_normal(mean_vec, cov_mat, sample_size)

    return reference_distribution, scaled_distributions

# Assume real and fake prior is equal
def getExponential(sample_size, dimension, lambda_factors):
    reference_distribution = np.random.exponential(1 / lambda_factors[0], size=(sample_size, dimension))
    scaled_distributions = []
    for scale in lambda_factors:
        samples = np.random.exponential(1 / scale, size=(sample_size, dimension))
        scaled_distributions.append(samples)

    return reference_distribution, scaled_distributions

def getDensities(sample_size, dimension, lambda_factors, distribution_name="gaussian"):
    if distribution_name == "gaussian":
        return getGaussian(sample_size, dimension, lambda_factors)
    elif distribution_name == "exponential":
        return getExponential(sample_size, dimension, lambda_factors)

def saveDistributions(iterations, sample_sizes, dimensions, lambda_factors, save_path):
    distribution_dict = {}
    for iter in range(iterations):
        for sample_size in sample_sizes:
            for dimension in dimensions:
                for scale in lambda_factors:
                    key = (iter, sample_size, dimension, scale)
                    mean_vec = np.zeros(dimension)
                    identity_cov = np.eye(dimension)*scale
                    samples = np.random.multivariate_normal(mean_vec, identity_cov, sample_size)
                    distribution_dict[key] = samples

    helper.savePickle(save_path, distribution_dict)
    return distribution_dict
