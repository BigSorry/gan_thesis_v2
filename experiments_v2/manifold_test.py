import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visualize as plotting
import experiments_v2.helper_functions as util
from sklearn.model_selection import KFold

def getDistributions(sample_sizes, dimensions, lambda_factors, data_iters):
    real_sets = []
    fake_sets = []
    params_used = []
    for iter in range(data_iters):
        for samples in sample_sizes:
            for dimension in dimensions:
                mean_vec = np.zeros(dimension)
                random_mean_add = np.random.normal(size=dimension)
                for scale_factor in lambda_factors:
                    mean_real = mean_vec + random_mean_add
                    cov_real = np.eye(dimension)
                    cov_fake = np.eye(dimension) * scale_factor

                    real_features = np.random.multivariate_normal(mean_real, cov_real, size=samples)
                    fake_features = np.random.multivariate_normal(mean_vec, cov_fake, size=samples)
                    real_sets.append(real_features)
                    fake_sets.append(fake_features)
                    params_used.append([iter, samples, dimension, scale_factor])

    return real_sets, fake_sets, params_used
def doEval(real_features, fake_features, k_params, circle_iters, percentage_off):
    columns = ["iter", "k_val", "recall", "precision"]
    row_data = []
    samples = real_features.shape[0]
    distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = util.getDistanceMatrices(real_features, fake_features)
    for k in k_params[samples]:
        boundaries_real = distance_matrix_real[:, k]
        boundaries_fake = distance_matrix_fake[:, k]
        off_samples = int(samples * percentage_off)
        for i in range(circle_iters):
            off_indices = np.random.choice(samples, off_samples, replace=False)
            boundaries_real_used = boundaries_real.copy()
            boundaries_real_used[off_indices] = 0
            boundaries_fake_used = boundaries_fake.copy()
            boundaries_fake_used[off_indices] = 0
            # Turn off fake samples for Coverage

            special_coverage = util.getCoverageSpecial(distance_matrix_pairs, boundaries_real, off_indices)
            # Turn off circles works only for Precision/Recall
            precision, recall, density, coverage = util.getScores(distance_matrix_pairs, boundaries_fake_used,
                                                                   boundaries_real_used, k)
            row = [i, k, recall, precision]
            row_data.append(row)

    dataframe = pd.DataFrame(columns=columns, data=row_data)
    return dataframe

def plotInfo(score_dataframe, data_params, percentage_off):
    samples = data_params[1]
    dimension = data_params[2]
    lambda_factor = data_params[3]
    experiment_key = f"{samples}_{dimension}_{lambda_factor}"
    grouped_data = score_dataframe.groupby("k_val")

    recalls = []
    precisions = []
    k_vals = []
    for k_val, group_data in grouped_data:
        recall = group_data["recall"].values
        precision = group_data["precision"].values
        recalls.append(recall)
        precisions.append(precision)
        k_vals.append(k_val)

    boxplot(recalls, precisions,  k_vals,
            f"Samples {samples} with dimension {dimension}, lambda factor {lambda_factor}, "
            f"and {percentage_off*100}% circles turned off",
            save=True, save_path=f"../fig_v2/recall/split/{experiment_key}.png")



def boxplot(score_diversity, score_fidelity, x_ticks, title_text, save, save_path):
    plt.figure()
    plt.suptitle(title_text)

    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title("Recall")
    ax1.set_xlabel("K value")
    ax1.set_ylim([0, 1.1])
    ax1.boxplot(score_diversity)
    ax1.set_xticks(np.arange(len(score_diversity)) + 1)
    ax1.set_xticklabels(x_ticks, rotation=90)

    ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
    ax2.set_title("Precision")
    ax2.boxplot(score_fidelity)
    ax2.set_xticks(np.arange(len(score_diversity)) + 1)
    ax2.set_xticklabels(x_ticks, rotation=90)

    if save:
        plt.subplots_adjust(wspace=0.3)
        plt.savefig(f"{save_path}",
                    dpi=300, bbox_inches='tight')
        plt.close()

def experimentManifold():
    # Get datasets
    data_iters = 10
    dimensions = [2, 32]
    sample_sizes = [1000]
    lambda_factors = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    lambda_factors = [0.001, 0.1, 1, 100, 1000]
    real_sets, fake_sets, params_used = getDistributions(sample_sizes, dimensions, lambda_factors, data_iters)
    # Evaluation procedure
    circle_off_iters = 10
    percentage_off = 0.0
    k_vals = {samples: util.getParams(samples) for samples in sample_sizes}
    print(k_vals)
    for i in range(len(real_sets)):
        real_features = real_sets[i]
        fake_features = fake_sets[i]
        data_params = params_used[i]
        score_dataframe = doEval(real_features, fake_features, k_vals, circle_off_iters, percentage_off)
        plotInfo(score_dataframe, data_params, percentage_off)

#experimentManifold()