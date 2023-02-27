import numpy as np
from scipy.stats import multivariate_normal

# Assume real and fake prior is equal
def multiGaus(real_data, fake_data, scale_params):
    mixture_data = np.concatenate([real_data, fake_data])
    predictions = np.zeros(mixture_data.shape[0])
    dim = mixture_data.shape[1]
    mean_vec = np.zeros(dim)
    cov_real = np.eye(dim) * scale_params[0]
    cov_fake = np.eye(dim) * scale_params[1]
    densities_real = multivariate_normal.pdf(mixture_data, mean=mean_vec, cov=cov_real)
    densities_fake = multivariate_normal.pdf(mixture_data, mean=mean_vec, cov=cov_fake)

    return densities_real, densities_fake
def multiUniform(real_data, fake_data, scale_params):
    width = np.abs(scale_params[1] - scale_params[0])

def getDensities(real_data, fake_data, distribution_parameters, method_name="multi_gaus"):
    if method_name == "multi_gaus":
        return multiGaus(real_data, fake_data, distribution_parameters)