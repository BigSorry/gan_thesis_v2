from data import gaus_problem as problem
import numpy as np
import visualize as plotting
from scipy.stats import multivariate_normal


def getPRLambdas(angle_count = 50):
    epsilon = 1e-10
    angles = np.linspace(epsilon, np.pi / 2 - epsilon, num=angle_count)
    lambdas = np.tan(angles)

    return lambdas

def getMixture(real, fake):
    mixture = np.zeros(real.shape)
    labels = np.zeros(real.shape[0])
    for i in range(real.shape[0]):
        flip_coin = np.random.randint(2)
        if flip_coin == 0:
            mixture[i, :] = fake[i, :]
        else:
            mixture[i, :] = real[i, :]
            labels[i] = 1

    return mixture, labels

def getScores(truth, predictions):
    # Null hypothesis mixture data is from real
    # FPR false rejection null
    # FNR failure to reject null
    real_label = 1
    fake_label = 0
    epsilon = 1e-10
    real_samples = np.sum(truth == real_label)
    fake_samples = np.sum(truth == fake_label)
    fp =  np.sum((predictions==fake_label) & (truth==real_label))
    fn =  np.sum((predictions==real_label) & (truth==fake_label))

    fpr = fp / (real_samples+epsilon)
    fnr = fn / (fake_samples+epsilon)

    return fpr, fnr

# TODO Refactor

def getExponential(mixture_data, real_param, fake_param, threshold):
    predictions = np.zeros(mixture_data.shape[0])
    dimension = mixture_data.shape[1]
    densities_real = 1
    densities_fake = 1
    for i in range(dimension):
        densities_real *= (real_param * np.exp(-real_param * mixture_data[:, i])).flatten()
        densities_fake *= (fake_param * np.exp(-fake_param * mixture_data[:, i])).flatten()
    predictions[threshold*densities_real >= densities_fake] = 1

    return predictions
def getPredictions(mixture_data, real_params, fake_params, threshold, distribution_name=""):
    predictions = np.zeros(mixture_data.shape[0])
    # Mixture problem
    if distribution_name == "exp":
        predictions = getExponential(mixture_data, real_params, fake_params, threshold)
        return predictions
    else:
        dim = mixture_data.shape[1]
        mean_vec = np.zeros(dim)
        cov_real = np.eye(dim)*real_params
        cov_fake = np.eye(dim)*fake_params
        densities_real = multivariate_normal.pdf(mixture_data, mean=mean_vec, cov=cov_real)
        densities_fake = multivariate_normal.pdf(mixture_data, mean=mean_vec, cov=cov_fake)

    predictions[threshold*densities_real >= densities_fake] = 1

    return predictions

def getPRCurve(mixture_samples, labels, lambdas,
               real_params, fake_params, distribution_name=""):
    curve = np.zeros((lambdas.shape[0], 2))
    differences = []
    for row_index, lambda_val in enumerate(lambdas):
        predictions = getPredictions(mixture_samples, real_params, fake_params, lambda_val, distribution_name)
        fpr, fnr = getScores(labels, predictions)
        precision = (fpr * lambda_val) + fnr
        recall = precision / lambda_val
        curve[row_index, 0] = precision
        curve[row_index, 1] = recall

    return np.clip(curve, 0, 1), differences

# TODO
def getPRCurveTest(mixture_samples, labels, lambdas,
               real_params, fake_params, distribution_name=""):
    curve = np.zeros((lambdas.shape[0], 2))
    curve2 = np.zeros((lambdas.shape[0], 2))
    differences = []
    for row_index, lambda_val in enumerate(lambdas):
        predictions = getPredictions(mixture_samples, real_params, fake_params, lambda_val, distribution_name)
        fpr, fnr = getScores(labels, predictions)
        precision = (fpr * lambda_val) + fnr
        recall = precision / lambda_val
        curve[row_index, 0] = fpr
        curve[row_index, 1] = fnr
        curve2[row_index, 0] = precision
        curve2[row_index, 1] = recall

    return np.clip(curve, 0, 1), np.clip(curve2, 0, 1)

    # Mixture problem Backup
    # elif len(real_params) == 3:
    #     modes = real_params[0].shape[0]
    #     mode_weight = real_params[2]
    #     mode_weight_fake = fake_params[2]
    #     densities_real = np.zeros(mixture_data.shape[0])
    #     densities_fake = np.zeros(mixture_data.shape[0])
    #     for i in range(modes):
    #         mean_vector = real_params[0][i, :]
    #         mean_vector_fake = fake_params[0][i, :]
    #         # Assumes covariance stays the same
    #         covariance = real_params[1]
    #         covariance_fake = fake_params[1]
    #
    #         densities_mode = multivariate_normal.pdf(mixture_data, mean=mean_vector, cov=covariance) * mode_weight[i]
    #         densities_fake_mode = multivariate_normal.pdf(mixture_data, mean=mean_vector_fake, cov=covariance_fake) * mode_weight_fake[i]
    #         densities_real += densities_mode
    #         densities_fake += densities_fake_mode
    #     densities_real = densities_real*threshold