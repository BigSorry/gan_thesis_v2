from data import gaus_problem as problem
import numpy as np
import visualize as plotting
from scipy.stats import multivariate_normal

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
    # A positive result corresponds to rejecting the null hypothesis
    # Null -> sample is from real data
    correct_mask = truth == predictions
    fp = truth[(correct_mask == False) & (truth == 0)].shape[0]
    positives = np.sum(truth == 0)

    fn = truth[(correct_mask == False) & (truth == 1)].shape[0]
    negatives = np.sum(truth == 1)

    fpr = fp / positives
    fnr = fn / negatives

    return fpr, fnr

def getPredictions(mixture_data, real_params, fake_params, threshold):
    predictions = np.zeros(mixture_data.shape[0])
    # Mixture problem
    if len(real_params) == 3:
        modes = real_params[0].shape[0]
        mode_weight = real_params[2]
        mode_weight_fake = fake_params[2]
        densities_real = np.zeros(mixture_data.shape[0])
        densities_fake = np.zeros(mixture_data.shape[0])
        for i in range(modes):
            mean_vector = real_params[0][i, :]
            mean_vector_fake = fake_params[0][i, :]
            # Assumes covariance stays the same
            covariance = real_params[1]
            covariance_fake = fake_params[1]

            densities_mode = multivariate_normal.pdf(mixture_data, mean=mean_vector, cov=covariance) * mode_weight[i]
            densities_fake_mode = multivariate_normal.pdf(mixture_data, mean=mean_vector_fake, cov=covariance_fake) * mode_weight_fake[i]
            densities_real += densities_mode
            densities_fake += densities_fake_mode
        densities_real = densities_real*threshold

    else:
        densities_real = multivariate_normal.pdf(mixture_data, mean=real_params[0], cov=real_params[1])*threshold
        densities_fake = multivariate_normal.pdf(mixture_data, mean=fake_params[0], cov=fake_params[1])

    predictions[densities_real >= densities_fake] = 1

    return predictions

def getPRCurve(mixture_samples, labels, lambdas,
               real_params, fake_params):
    curve = np.zeros((lambdas.shape[0], 2))

    # For boundary visualization in 2D
    special_lambdas = [0, 0.5, 1, 1.5, 2]
    real_samples = mixture_samples[labels == 1, :]
    fake_samples = mixture_samples[labels == 0, :]
    for row_index, lambda_val in enumerate(lambdas):
        predictions = getPredictions(mixture_samples, real_params, fake_params, lambda_val)
        fpr, fnr = getScores(labels, predictions)

        boolean_mask = np.isclose(lambda_val, special_lambdas)
        if np.any(boolean_mask):
            special_lambdas.pop(0)
            plotting.showLabels(real_samples, fake_samples, mixture_samples, predictions,
                                lambda_val, 1)
        precision = (fnr * lambda_val) + fpr
        recall = precision / lambda_val
        curve[row_index, 0] = precision
        curve[row_index, 1] = recall

    return np.clip(curve, 0, 1)

# TODO
def getPRCurveTest(mixture_samples, labels, lambdas,
               real_params, fake_params):
    curve = np.zeros((lambdas.shape[0], 2))
    errorRates = []
    predictions = getPredictions(mixture_samples, real_params, fake_params, 1)
    for row_index, lambda_val in enumerate(lambdas):
        predictions = getPredictions(mixture_samples, real_params, fake_params, lambda_val)
        fpr, fnr = getScores(labels, predictions)
        errorRates.append((float(fpr), float(fnr)))

    for row_index, lambda_val in enumerate(lambdas):
        precision = np.min([(lambda_val * fnr) + fpr for fpr, fnr in errorRates])
        recall = precision / lambda_val
        curve[row_index, 0] = precision
        curve[row_index, 1] = recall

    return np.clip(curve, 0, 1)