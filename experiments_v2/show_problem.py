import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visualize as plotting
import experiments_v2.helper_functions as util
from sklearn.model_selection import KFold

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
        # boxplot(coverage_splits, list(k_dict.keys()),
        #         f"Coverage, samples {experiment_key[0]} with dimension {experiment_key[1]} and lambda factor {experiment_key[2]}")


    plt.show()

def boxplot(scores, x_ticks, title_text):
    plt.figure()
    plt.title(title_text)
    plt.boxplot(scores)
    plt.ylim([0, 1.1])
    plt.xlabel("K Value")
    plt.xticks(np.arange(len(scores)) + 1, x_ticks, rotation=90)

def doEval(sample_size, dimension, k_params, lambda_factors, splits=5):
    mean_vec = np.zeros(dimension)
    for scale_factor in lambda_factors:
        mean_real = mean_vec
        cov_real = np.eye(dimension)
        cov_fake = np.eye(dimension) * scale_factor
        real_features = np.random.multivariate_normal(mean_real, cov_real, size=sample_size)
        fake_features = np.random.multivariate_normal(mean_vec, cov_fake, size=sample_size)
        distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = util.getDistanceMatrices(real_features, fake_features)
        for k in k_params:
            boundaries_real = distance_matrix_real[:, k]
            boundaries_fake = distance_matrix_fake[:, k]
            kf = KFold(n_splits=splits, random_state=None, shuffle=True)
            for train_index, test_index in kf.split(boundaries_real):
                boundaries_real_used = boundaries_real.copy()
                boundaries_real_used[test_index] = 0
                boundaries_fake_used = boundaries_fake.copy()
                boundaries_fake_used[test_index] = 0

                recall_mask, coverage_mask = util.getScoreMask(boundaries_real_used, boundaries_fake_used, distance_matrix_pairs)
                recall = (distance_matrix_pairs < np.expand_dims(boundaries_fake_used, axis=0)).any(axis=1).mean()

                title_text = f"Recall {recall}, with k {k} and splits {splits}"
                plottingRecall(real_features, fake_features, boundaries_fake_used,
                         recall_mask, title_text, False, "")


def plottingRecall(real_data, fake_data, boundaries_fake, recall_mask, title_text, save=False, save_path=""):
    # Start plotting
    plt.figure()
    plt.title(title_text)
    # Recall manifold
    plotting.plotCircles(fake_data, boundaries_fake)
    plt.scatter(fake_data[:, 0], fake_data[:, 1],
                label="Fake Samples", c="blue", s=2 ** 4, zorder=99, alpha=0.75)
    plotting.plotAcceptRejectData(real_data, recall_mask, data_kind="real")
    plt.legend()
    if save:
        plt.subplots_adjust(wspace=0.3)
        plt.savefig(f"{save_path}points.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

def experimentManifold():
    # Data Setup
    dimension = 2
    sample_size = 10
    k_vals = [1]
    lambda_factors = [100]
    doEval(sample_size, dimension, k_vals, lambda_factors, splits=10)

experimentManifold()
plt.show()