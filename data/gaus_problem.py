import numpy as np
import numpy.random
from scipy.stats import multivariate_normal
from scipy.stats import multivariate_normal

def getGauss(samples, params, dimension=3):
    mean = params[0]
    std = params[1]
    mean_vec = np.array([mean, mean])
    cov = np.array([[std, 0], [0, std]])
    real_features = np.random.multivariate_normal(mean_vec, cov, samples)
    real_features = np.random.normal(mean, std, size=(samples, dimension))

    return real_features

def getLikelihood(data, mean_val, sigma):
    mean = np.array([mean_val, mean_val])
    residuals = data - mean
    cov = np.array([[sigma, 0],
                   [0, sigma]])
    loglikelihood = -0.5 * (
            np.log(np.linalg.det(cov))
            + np.einsum('...j,jk,...k', residuals, np.linalg.inv(cov), residuals)
            + len(mean) * np.log(2 * np.pi)
    )
    return -np.sum(loglikelihood)

def testLike(samples, mean_val, cov_val):
    mean = np.array([mean_val, mean_val])
    cov = np.array([[cov_val, 0], [0, cov_val]])
    y = multivariate_normal.pdf(samples, mean=mean, cov=cov)

    return np.sum(y)

def getDensities(samples, mean_val, cov_val):
    mean = np.array([mean_val for i in range(samples.shape[1])])
    cov = np.zeros((samples.shape[1], samples.shape[1]))
    for i in range(samples.shape[1]):
        cov[i, i] = cov_val
    y = multivariate_normal.pdf(samples, mean=mean, cov=cov)

    return y

def genereateMixtureData(mode_sample_counts, dimension):
    cov_matrix = np.eye(dimension)
    modes = mode_sample_counts.shape[0]
    real_features = np.zeros(shape=(np.sum(mode_sample_counts, dtype=np.int), dimension))
    mean_vectors = np.random.random(size=(modes, dimension))
    for mode in range(modes):
        mean_vec = mean_vectors[mode, :] * mode
        real_samples = np.random.multivariate_normal(mean_vec, cov_matrix,
                                                     size=real_mode_counts[mode])
        begin_index = np.sum(real_mode_counts[:mode], dtype=np.int)
        end_index = begin_index + np.sum(real_mode_counts[mode], dtype=np.int)
        real_features[begin_index:end_index, :] = real_samples

    return real_features

dims=[64]
mode_samples = 500
modes = 4
real_mode_counts = np.array([mode_samples for i in range(modes)])
real_features = genereateMixtureData(real_mode_counts, dims[0])