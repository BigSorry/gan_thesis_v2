import numpy as np
import pandas as pd
import experiments_v2.helper_functions as util
import matplotlib.pyplot as plt
import metrics.likelihood_classifier as llc
from sklearn.metrics import pairwise_distances

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

def doExpiriment():
    sample_size = 2000
    dimension = 2
    scale_factors = [0.01, 1]
    k_vals = [1, 2, 4, 8, 16, 32, sample_size - 1]
    distributions = getDistributions(sample_size, dimension, scale_factors)
    test_distribution = getDistributions(sample_size, dimension, scale_factors)
    real_data = distributions[0]
    fake_data = distributions[1]
    mixture_dict = doMixture(real_data, fake_data)
    errors_real = doKNNS(mixture_dict, real_data, k_vals, label_value=1)
    errors_fake = doKNNS(mixture_dict, fake_data, k_vals, label_value=0)
    doPlotting(errors_real)
    doPlotting(errors_fake)

doExpiriment()
plt.show()