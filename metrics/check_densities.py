import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from experiments import create_experiment as exp
import experiments.experiment_visualization as exp_vis
from sklearn import metrics

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

def doCheck(dimensions, factors):
    iters = 1
    sample_size = 1000
    k_vals = [i for i in range(1, 1000, 10)]
    real_mean_vectors = []
    fake_mean_vectors = []
    for i in range(iters):
        for dimension in dimensions:
            for scale in factors:
                lambda_factors = [factors[0], scale]
                reference_distribution, scaled_distribution = getGaussian(sample_size, dimension, scale)
                densities_real, densities_fake = multiGaus(reference_distribution, scaled_distribution, dimension, lambda_factors)
                curve_classifier, curve_var_dist = exp.getGroundTruth("gaussian", reference_distribution,  scaled_distribution, lambda_factors)

                auc = 0
                #auc = metrics.auc(np.round(curve_var_dist[:, 1], 2), np.round(curve_var_dist[:, 0], 2))

                if auc < 0.95 and auc > 0.05 or 1==1:
                    plt.figure()
                    plt.title(lambda_factors)
                    #exp_vis.plotTheoreticalCurve(curve_var_dist, curve_var_dist, lambda_factors, save=False)
                    pr_pairs, dc_pairs = exp.getKNNData(reference_distribution, scaled_distribution,  k_vals)
                    exp_vis.plotCurve(curve_classifier,  "classifier")
                    plt.figure()
                    plt.title(lambda_factors)
                    exp_vis.plotCurve(curve_var_dist, "var dist")

                avg_ll_fake = np.mean(densities_real)
                avg_ll_real = np.mean(densities_fake)
                real_mean_vectors.append(avg_ll_real)

    real_mean = np.mean(np.array(real_mean_vectors))
    real_std = np.mean(np.std(real_mean_vectors))
    print(real_mean, real_std)

def tryValues():
    base_values = np.linspace(0.11, 0.05, 10)
    print(base_values)
    dimensions = [16]

    for value in base_values:
        used_value = np.round(value, 4)
        factors = [used_value * 1.1 ** (-i) for i in range(5)]
        factors = np.round(factors, 4)
        doCheck(dimensions, factors)


# dimensions = [2]
# factors = [1 * 2 ** (-i) for i in range(8)]
# factors = np.round(factors, 2)
# doCheck(dimensions, factors)


dimensions = [2]
factors = [1*2** (-i) for i in range(5)]
factors = np.round(factors, 5)
doCheck(dimensions, factors)
plt.show()
