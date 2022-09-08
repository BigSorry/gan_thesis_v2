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

def getData(iters, dimension, sample_sizes, lambda_factors):
    columns = ["iter", "sample_size", "dimension", "lambda",
               "k_val", "recall", "coverage"]
    row_data = []
    mean = np.zeros(dimension)
    for iter in range(iters):
        for samples in sample_sizes:
            k_vals = getKParams(samples, max_k=int(samples), step_size=20)
            print(k_vals)
            for scale_factor in lambda_factors:
                cov_real = np.eye(dimension)
                cov_fake = np.eye(dimension) * scale_factor
                real_features = np.random.multivariate_normal(mean, cov_real, size=samples)
                fake_features = np.random.multivariate_normal(mean, cov_fake, size=samples)
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

def plotExperiment(dataframe, sample_size, show_boxplot,
                   show_map, path_box, path_map):
        if show_boxplot:
            grouped_data = dataframe.groupby(["k_val"])
            recalls = []
            coverages = []
            xticks = []
            for k_val, group_data in grouped_data:
                recall = group_data["recall"].values
                coverage = group_data["coverage"].values
                recalls.append(recall)
                coverages.append(coverage)
                xticks.append(k_val)

            plotting.plotBox(recalls, xticks, f"Recall with sample size {sample_size}",
                             save=True, save_path=path_box+f"recall_samples{sample_size}.png")
            plotting.plotBox(coverages, xticks, f"Coverages with sample size {sample_size}",
                             save=True, save_path=path_box+f"coverage_samples{sample_size}.png")

        if show_map:
            recall_pivot = pd.pivot_table(dataframe, values='recall', index=['lambda'],
                    columns=['k_val'], aggfunc=np.mean)
            coverage_pivot = pd.pivot_table(dataframe, values='coverage', index=['lambda'],
                                          columns=['k_val'], aggfunc=np.mean)

            plotting.HeatMapPivot(recall_pivot, title_text=f"Recall with sample size {sample_size}",
                                  save=True, save_path=path_map+f"recall_samples{sample_size}.png")
            plotting.HeatMapPivot(coverage_pivot, title_text=f"Coverage with sample size {sample_size}",
                                  save=True, save_path=path_map+f"coverage_samples{sample_size}.png")


import seaborn as sns
def sampleExperiment():
    # Data Setup
    iters = 2
    dimension = 64
    sample_sizes = [500, 1000]
    lambda_factors = [0.01, 0.1, 10, 100]
    dataframe = getData(iters, dimension, sample_sizes, lambda_factors)

    show_box=True
    show_map=True
    # PC paths
    path_map = "C:/Users/Lex/OneDrive/plots_thesis/pc/heatmap/"
    path_box = "C:/Users/Lex/OneDrive/plots_thesis/pc/boxplot/"
    path_map = "C:/Users/lexme/OneDrive/plots_thesis/laptop/heatmap/"
    path_box = "C:/Users/lexme/OneDrive/plots_thesis/laptop/boxplot/"
    for sample_size in sample_sizes:
            select_data = dataframe.loc[dataframe["sample_size"] == sample_size, :]
            plotExperiment(select_data, sample_size, show_box,
                           show_map, path_box, path_map)


