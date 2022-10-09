import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visualize as plotting
import experiments_v2.helper_functions as util
from sklearn.model_selection import KFold

def doEval(sample_sizes, dimensions, k_params, lambda_factors, iters, splits=5):
    columns = ["iter", "sample_size", "dimension", "lambda",
               "k_val", "recall", "coverage"]
    row_data = []
    for iter in range(iters):
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
                    for k in k_params[samples]:
                        boundaries_real = distance_matrix_real[:, k]
                        boundaries_fake = distance_matrix_fake[:, k]
                        kf = KFold(n_splits=splits, random_state=None, shuffle=True)
                        #for train_index, test_index in kf.split(boundaries_real):
                        indices = [i for i in range(samples)]
                        off_samples = 995
                        for i in range(50):
                            off_indices = np.random.choice(samples, off_samples, replace=False)
                            boundaries_real_used = boundaries_real.copy()
                            boundaries_real_used[off_indices] = 0
                            boundaries_fake_used = boundaries_fake.copy()
                            boundaries_fake_used[off_indices] = 0
                            # Turn off fake samples for Coverage
                            special_coverage = util.getCoverageSpecial(distance_matrix_pairs, boundaries_real, off_indices)
                            # Turn off only fake circles for Recall
                            metric_scores = util.getScores(distance_matrix_pairs, boundaries_fake_used,
                                                                                   boundaries_real, k)
                            row = [iter, samples, dimension, scale_factor, k, metric_scores[1], special_coverage]
                            row_data.append(row)

    dataframe = pd.DataFrame(columns=columns, data=row_data)
    return dataframe

def plotInfo(dataframe):
    experiment_group = dataframe.groupby(["sample_size", "dimension", "lambda"])
    for experiment_key, experiment_group in experiment_group:
        recall_data = []
        k_vals = experiment_group["k_val"].unique()
        for k in k_vals:
            recalls = experiment_group.loc[experiment_group["k_val"] == k, "recall"].values
            recall_data.append(recalls)

        boxplot(recall_data, k_vals,
                f"Recall, samples {experiment_key[0]} with dimension {experiment_key[1]}, lambda factor {experiment_key[2]}, and 10 splits",
                save=True, save_path=f"../fig_v2/recall/kfold/{experiment_key}.png")
        # boxplot(coverage_splits, list(k_dict.keys()),
        #          f"Coverage, samples {experiment_key[0]} with dimension {experiment_key[1]} and lambda factor {experiment_key[2]}")


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
    # Data Setup
    iters = 10
    dimensions = [2, 32, 64]
    sample_sizes = [1000]
    k_vals = {samples:getParams(samples) for samples in sample_sizes}
    print(k_vals)
    lambda_factors = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    lambda_factors = [1000]
    dataframe = doEval(sample_sizes, dimensions, k_vals, lambda_factors, iters, splits=10)
    plotInfo(dataframe)

experimentManifold()