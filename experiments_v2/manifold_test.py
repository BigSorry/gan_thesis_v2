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
    columns = ["iter", "k_val", "recall"]
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
            row = [i, k, recall]
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
    k_vals = []
    for k_val, group_data in grouped_data:
        recall = group_data["recall"].values
        recalls.append(recall)
        k_vals.append(k_val)

    boxplot(recalls, k_vals,
            f"Recall, samples {samples} with dimension {dimension}, lambda factor {lambda_factor}, "
            f"and {percentage_off*100}% circles turned off",
            save=True, save_path=f"../fig_v2/recall/split/{experiment_key}.png")
    # boxplot(coverages, k_vals,
    #           f"Precision, samples {samples]} with dimension {dimension}, lambda factor {lambda_factor}, "
    #         f"and {percentage_off*100}% circles turned off",
    #         save=True, save_path=f"../fig_v2/precision/split/{experiment_key}.png")


def boxplot(scores, x_ticks, title_text, save, save_path):
    plt.figure()
    plt.title(title_text)
    plt.boxplot(scores)
    plt.ylim([0, 1.1])
    plt.xlabel("K Value")
    plt.xticks(np.arange(len(scores)) + 1, x_ticks, rotation=90)
    if save:
        plt.subplots_adjust(wspace=0.3)
        plt.savefig(f"{save_path}",
                    dpi=300, bbox_inches='tight')
        plt.close()

def getParams(sample_size):
    k_values = []
    for i in range(sample_size):
        k_val = 2 **i
        if k_val > sample_size:
            break
        else:
            k_values.append(k_val)
    k_values.append(sample_size - 1)

    return k_values

def experimentManifold():
    # Parameters setup
    data_iters = 10
    circle_iters = 10
    percentage_off = 0.95
    dimensions = [32]
    sample_sizes = [1000]
    lambda_factors = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    lambda_factors = [1, 1000]
    k_vals = {samples:getParams(samples) for samples in sample_sizes}
    print(k_vals)
    # Get data
    real_sets, fake_sets, params_used = getDistributions(sample_sizes, dimensions, lambda_factors, data_iters)
    # Evaluation procedure
    for i in range(len(real_sets)):
        real_features = real_sets[i]
        fake_features = fake_sets[i]
        data_params = params_used[i]
        score_dataframe = doEval(real_features, fake_features, k_vals, circle_iters, percentage_off)
        plotInfo(score_dataframe, data_params, percentage_off)

experimentManifold()