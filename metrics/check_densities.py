import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from experiments import create_experiment as exp
import experiments.experiment_visualization as exp_vis

# Assume real and fake prior is equal
def multiGaus(real_data, fake_data, dimension, scale_params):
    mixture_data = np.concatenate([real_data, fake_data])
    mean_vec = np.zeros(dimension)
    cov_real = np.eye(dimension) * scale_params[0]
    cov_fake = np.eye(dimension) * scale_params[1]
    densities_real = multivariate_normal.pdf(mixture_data, mean=mean_vec, cov=cov_real)
    densities_fake = multivariate_normal.pdf(mixture_data, mean=mean_vec, cov=cov_fake)


    return densities_real, densities_fake

def getGaussian(sample_size, dimension, other_scale):
    mean_vec = np.zeros(dimension)
    identity_cov = np.eye(dimension)
    reference_distribution = np.random.multivariate_normal(mean_vec, identity_cov, sample_size)
    cov_mat = identity_cov*other_scale
    scaled_distributions = np.random.multivariate_normal(mean_vec, cov_mat, sample_size)

    return reference_distribution, scaled_distributions


def doCheck():
    iters = 5
    sample_size = 10000
    dimensions = [1000]
    factors = [.09*1.1 ** (-i) for i in range(5)]
    factors = np.round(factors, 4)
    print(factors)
    for i in range(iters):
        for dimension in dimensions:
            for scale in factors:
                lambda_factors = [factors[0], scale]
                reference_distribution, scaled_distributions = getGaussian(sample_size, dimension, scale)
                densities_real, densities_fake = multiGaus(reference_distribution, scaled_distributions, dimension, lambda_factors)
                curve_classifier, curve_var_dist = exp.getGroundTruth("gaussian", reference_distribution,
                                                                      scaled_distributions, lambda_factors)
                plt.figure()
                plt.title(lambda_factors)
                exp_vis.plotTheoreticalCurve(curve_var_dist, curve_var_dist, lambda_factors, save=False)
                real_sum = np.sum(densities_real)
                fake_sum = np.sum(densities_fake)

                if real_sum > 0.01 or fake_sum > 0.01:
                    print(dimension, scale)
                    print(np.sum(densities_real))
                    print(np.sum(densities_fake))
                    print()


doCheck()
plt.show()