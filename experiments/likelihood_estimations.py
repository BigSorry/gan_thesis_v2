import numpy as np
from scipy.stats import multivariate_normal

# Assume real and fake prior is equal
def multiGaus(mixture_data, dimension, scale_params):
    mean_vec = np.zeros(dimension)
    cov_real = np.eye(dimension) * scale_params[0]
    cov_fake = np.eye(dimension) * scale_params[1]
    densities_real = multivariate_normal.pdf(mixture_data, mean=mean_vec, cov=cov_real)
    densities_fake = multivariate_normal.pdf(mixture_data, mean=mean_vec, cov=cov_fake)
    densities_real = densities_real / np.sum(densities_real)
    densities_fake = densities_fake / np.sum(densities_fake)

    return densities_real, densities_fake

def multiExponential(mixture_data, dimension, scale_params):
    densities_real = np.zeros(mixture_data.shape[0])
    densities_fake = np.zeros(mixture_data.shape[0])
    real_lambda = scale_params[0]
    fake_lambda = scale_params[1]
    for col_index in range(dimension):
        densities_real += real_lambda*np.exp(-real_lambda*mixture_data[:, col_index])
        densities_fake += fake_lambda*np.exp(-fake_lambda*mixture_data[:, col_index])

    densities_real = densities_real / np.sum(densities_real)
    densities_fake = densities_fake / np.sum(densities_fake)


    return densities_real, densities_fake

def getDensities(real_data, fake_data, distribution_parameters, method_name="gaussian"):
    mixture_data = np.concatenate([real_data, fake_data])
    dimension = mixture_data.shape[1]
    if method_name == "gaussian":
        return multiGaus(mixture_data, dimension, distribution_parameters)
    elif method_name == "exponential":
        return multiExponential(mixture_data, dimension, distribution_parameters)