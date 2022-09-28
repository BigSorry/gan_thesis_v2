import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visualize as plotting
import experiments_v2.helper_functions as util

def getKParams(sample_size, max_k, step_size=10):
    vals = []
    for i in range(1, max_k, step_size):
        size = i
        vals.append(size)
    if vals[len(vals) - 1] < (sample_size - 1):
        vals.append(sample_size-1)
    return vals

def getData(iters, dimensions, sample_sizes, lambda_factors):
    columns = ["iter", "sample_size", "dimension", "lambda",
               "k_val", "recall", "coverage"]
    row_data = []
    for iter in range(iters):
        for samples in sample_sizes:
            k_vals = getKParams(samples, max_k=int(samples), step_size=20)
            print(k_vals)
            for dimension in dimensions:
                mean_vec = np.zeros(dimension)
                for scale_factor in lambda_factors:
                    cov_real = np.eye(dimension)
                    cov_fake = np.eye(dimension) * scale_factor
                    real_features = np.random.multivariate_normal(mean_vec, cov_real, size=samples)
                    fake_features = np.random.multivariate_normal(mean_vec, cov_fake, size=samples)
                    distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = util.getDistanceMatrices(
                        real_features, fake_features)
                    for k_val in k_vals:
                        # Calculations
                        boundaries_real = distance_matrix_real[:, k_val]
                        boundaries_fake = distance_matrix_real[:, k_val]
                        precision, recall, density, coverage = util.getScores(distance_matrix_pairs, boundaries_fake,
                                                                              boundaries_real, k_val)
                        row = [iter, samples, dimension, scale_factor, k_val, recall, coverage]
                        row_data.append(row)

    dataframe = pd.DataFrame(columns=columns, data=row_data)
    return dataframe

from sklearn.model_selection import KFold
def doEval(sample_sizes, dimensions, k_params, lambda_factors, splits=5):
    info_dict = {}
    for samples in sample_sizes:
        for dimension in dimensions:
            mean_vec = np.zeros(dimension)
            for scale_factor in lambda_factors:
                mean_real = mean_vec
                cov_real = np.eye(dimension)
                cov_fake = np.eye(dimension) * scale_factor
                real_features = np.random.multivariate_normal(mean_real, cov_real, size=samples)
                fake_features = np.random.multivariate_normal(mean_vec, cov_fake, size=samples)
                distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = util.getDistanceMatrices(real_features, fake_features)
                key = (samples, dimension, scale_factor)
                if key not in info_dict:
                    info_dict[key] = {}
                for k in k_params[samples]:
                    boundaries_real = distance_matrix_real[:, k]
                    boundaries_fake = distance_matrix_fake[:, k]
                    kf = KFold(n_splits=splits, random_state=None, shuffle=True)
                    scores = []
                    for train_index, test_index in kf.split(boundaries_real):
                        boundaries_real_used = boundaries_real
                        boundaries_real_used[test_index] = 0
                        boundaries_fake_used = boundaries_fake
                        boundaries_fake_used[test_index] = 0
                        metric_scores = util.getScores(distance_matrix_pairs, boundaries_real_used,
                                                                               boundaries_fake_used, k)
                        scores.append(metric_scores)

                    info_dict[key][k] = np.array(scores)

    return info_dict

def plotInfo(info_dict):
    for experiment_key, k_dict in info_dict.items():
        recall_splits = []
        coverage_splits = []
        for index, (k_val, split_scores) in enumerate(k_dict.items()):
            recalls = split_scores[:, 1]
            coverage = split_scores[:, 3]
            recall_splits.append(recalls)
            coverage_splits.append(coverage)

        boxplot(recall_splits, list(k_dict.keys()),
                f"Recall, samples {experiment_key[0]} with dimension {experiment_key[1]} and lambda factor {experiment_key[2]}")
        boxplot(coverage_splits, list(k_dict.keys()),
                f"Coverage, samples {experiment_key[0]} with dimension {experiment_key[1]} and lambda factor {experiment_key[2]}")


    plt.show()

def boxplot(scores, x_ticks, title_text):
    plt.figure()
    plt.title(title_text)
    plt.boxplot(scores)
    plt.ylim([0, 1.1])
    plt.xlabel("K Value")
    plt.xticks(np.arange(len(scores)) + 1, x_ticks, rotation=90)

def experimentManifold():
    # Data Setup
    iters = 2
    dimensions = [2]
    sample_sizes = [1000]
    k_samples = 10
    k_vals = {samples:[2**i for i in range(k_samples)] for samples in sample_sizes}
    k_vals[1000].append(999)
    lambda_factors = [0.001, 1, 1000]
    info_dict = doEval(sample_sizes, dimensions, k_vals, lambda_factors, splits=100)
    plotInfo(info_dict)

experimentManifold()