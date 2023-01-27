import numpy as np
import pandas as pd
import experiments_v2.helper_functions as util
import matplotlib.pyplot as plt
import metrics.likelihood_classifier as llc
from sklearn.metrics import pairwise_distances
import experiments_v2.curve.likelihood_estimations as ll_est
import visualize as plotting
import experiments_v2.helper_functions as util
def getDistributions(sample_size, dimension, scale_param):
    distributions = []
    mean_vec = np.zeros(dimension)
    for scale in scale_param:
        cov_mat = np.eye(dimension)*scale
        samples = np.random.multivariate_normal(mean_vec, cov_mat, sample_size)
        distributions.append(samples)
    return distributions

def doKNNS(mixture_dict, test_data, k_vals, label_value):
    test_labels = np.array([label_value for i in range(test_data.shape[0])])
    error_rates = {}
    for ber_p, data_dict in mixture_dict.items():
        mixture_data = data_dict["train_data"]
        mixture_label = data_dict["train_label"]
        distance_matrix = util.getDistanceMatrix(mixture_data, mixture_data)
        distance_pairs = util.getDistanceMatrix(test_data, mixture_data)
        error_rates[ber_p] = {}
        for k_val in k_vals:
            boundaries_union = distance_matrix[:, k_val]
            row_form = np.expand_dims(boundaries_union, axis=0)
            truth_table = (distance_pairs < row_form)
            predictions = truth_table.any(axis=1).astype(int)
            fpr, fnr = llc.getScores(test_labels, predictions)
            error_rates[ber_p][k_val] = [fpr, fnr]

    return error_rates

def takeSamples(data, indices, p):
    sample_size = data.shape[0]
    taken_indices = int(sample_size * p)
    row_indices = np.random.choice(indices, taken_indices, replace=False)
    return data[row_indices, :]

def doMixture(real_data, fake_data):
    bernoulli_p = [0, 0.25, 0.5, 0.75, 1]
    indices = np.arange(real_data.shape[0])
    mixture_dict = {}
    for p in bernoulli_p:
        real_taken = takeSamples(real_data, indices, p)
        fake_taken = takeSamples(fake_data, indices, 1-p)
        mixture_data = np.concatenate([real_taken, fake_taken])
        mixture_labels = np.concatenate([np.ones(real_taken.shape[0]), np.zeros(fake_taken.shape[0])])
        mixture_dict[p] = {"train_data":mixture_data, "train_label":mixture_labels}

    return mixture_dict

def doPlotting(error_dict):
    fig, (ax1,ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    plt.title("Mixture Z= p.R + (1-p).F")
    ber_ps = list(error_dict.keys())
    plt.ylim([0, 1.1])
    x_axes = (np.arange(len(ber_ps))+1)*2
    i=2
    for ber_p, ber_dict in error_dict.items():
        scores = np.array(list(ber_dict.values()))
        fpr = scores[:, 0]
        fnr = scores[:, 1]
        ax1.boxplot(fpr, positions=[i])
        ax2.boxplot(fnr, positions=[i])
        i+=2


    ax1.set_ylabel("FPR")
    string_labels = [f"p={p}" for p in ber_ps]
    plt.xticks(x_axes, string_labels)
    ax1.set_xticks(x_axes)
    ax1.set_xticklabels(string_labels)
    ax1.set_xlim(0, np.max(x_axes)+2)


    ax2.set_ylabel("FNR")

def plotCurve(curve_classifier, curve_var_dist):
    plotting.plotCurve(curve_classifier, label_text="Likelihood ratio test")
    plotting.plotCurve(curve_var_dist, label_text="Variational distance")
    plt.legend()


def getGroundTruth(real_data, fake_data, scale_factors):
    lambdas = llc.getPRLambdas(angle_count=1000)
    densities_real, densities_fake = ll_est.getDensities(real_data, fake_data, scale_factors, method_name="multi_gaus")
    densities_real_norm = densities_real / np.sum(densities_real)
    densities_fake_norm = densities_fake / np.sum(densities_fake)

    curve_class = []
    curve_distance = []
    for value in lambdas:
        predictions = (value*densities_real >= densities_fake).astype(int)
        precision_hist = value*1 - np.sum(np.abs(value*densities_real_norm - densities_fake_norm))
        recall_hist = (1/value)*1 - np.sum(np.abs(densities_real_norm - densities_fake_norm*(1/value)))
        truth_labels = np.concatenate([np.ones(real_data.shape[0]), np.zeros(fake_data.shape[0])])
        fpr, fnr = llc.getScores(truth_labels, predictions)
        precision_class = value*fpr + fnr
        recall_class = precision_class / value
        curve_class.append([precision_class, recall_class])
        curve_distance.append([precision_hist, recall_hist])

    return np.array(curve_class), np.array(curve_distance)


def showGroundTruth(real_data, fake_data, scale_factors):
    curve_classifier, curve_var_dist = getGroundTruth(real_data, fake_data, scale_factors)
    plotCurve(curve_classifier, curve_var_dist)

def showKNN(real_data, fake_data, k_vals):
    distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = util.getDistanceMatrices(real_data, fake_data)
    #plt.figure()
    for k_val in k_vals:
        boundaries_real = distance_matrix_real[:, k_val]
        boundaries_fake = distance_matrix_fake[:, k_val]
        precision, recall, density, coverage = util.getScores(distance_matrix_pairs, boundaries_fake, boundaries_real, k_val)
        plt.scatter(recall, precision, c="red", label=f"Precision_Recall_k{k_val}")
        plt.scatter(precision, density, c="yellow", label=f"Density_Coverage_{k_val}")


def doExpiriment():
    sample_size = 1000
    dimension = 2
    var_factors = [0.1, 0.25, 0.5, 0.75, 1, 10, 100]
    for factor in var_factors:
        scale_factors = [1, factor]
        distributions = getDistributions(sample_size, dimension, scale_factors)
        real_data = distributions[0]
        fake_data = distributions[1]

        plt.figure()
        plt.title(f"Lambda scaling real cov {scale_factors[0]} and lambda scaling fake cov {scale_factors[1]}")
        showGroundTruth(real_data, fake_data, scale_factors)
        k_vals = [1, 2, 4, 8, 16, 32, sample_size - 1]
        k_vals = np.arange(1, sample_size, 5)
        print(k_vals)
        showKNN(real_data, fake_data, k_vals)


doExpiriment()
plt.show()