import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visualize as plotting
import experiments_v2.helper_functions as util

def getKParams(sample_size):
    vals = []
    for i in range(1, 10):
        size = int(sample_size * (i/10))
        vals.append(size)
    if vals[len(vals) - 1] < (sample_size - 1):
        vals.append(sample_size-1)
    return vals

import seaborn as sns
def sampleExperiment():
    # Setup
    iters = 2
    dimension = 2
    sample_sizes = [100, 1000]
    scale_factors = [0.01, 0.1, 1, 10, 1000]
    columns = ["iter", "sample_size", "dimension", "lambda", "k_val", "recall", "coverage"]
    row_data = []
    mean = np.zeros(dimension)
    for iter in range(iters):
        for samples in sample_sizes:
            k_vals = getKParams(samples)
            print(k_vals)
            for scale_factor in scale_factors:
                cov_real = np.eye(dimension)
                cov_fake = np.eye(dimension) * scale_factor
                real_features = np.random.multivariate_normal(mean, cov_real, size=samples)
                fake_features = np.random.multivariate_normal(mean, cov_fake, size=samples)
                distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = util.getDistanceMatrices(real_features, fake_features)
                for k_val in k_vals:
                    # Calculations
                    boundaries_real = distance_matrix_real[:, k_val]
                    boundaries_fake = distance_matrix_real[:, k_val]
                    precision, recall, density, coverage = util.getScores(distance_matrix_pairs, boundaries_fake, boundaries_real, k_val)
                    row = [iter, samples, dimension, scale_factor, k_val, recall, coverage]
                    row_data.append(row)

    datafame = pd.DataFrame(columns=columns, data=row_data)

    show_box=True
    show_map=True
    # PC paths
    path_map = "C:/Users/Lex/OneDrive/plots_thesis/pc/heatmap/"
    path_box = "C:/Users/Lex/OneDrive/plots_thesis/pc/boxplot/"
    for samples in sample_sizes:
            select_data = datafame.loc[datafame["sample_size"] == samples, :]
            if show_box:
                grouped_data = select_data.groupby(["k_val"])
                recalls = []
                coverages = []
                for name, group in grouped_data:
                    recall = group["recall"].values
                    coverage = group["coverage"].values
                    recalls.append(recall)
                    coverages.append(coverage)
                plotting.plotBox(recalls, f"Recall with sample size {samples}",
                                 save=True, save_path=path_box+f"recall_samples{samples}.png")
                plotting.plotBox(coverages, f"Coverages with sample size {samples}",
                                 save=True, save_path=path_box+f"coverage_samples{samples}.png")

            if show_map:
                recall_pivot = pd.pivot_table(select_data, values='recall', index=['lambda'],
                        columns=['k_val'], aggfunc=np.mean)
                coverage_pivot = pd.pivot_table(select_data, values='coverage', index=['lambda'],
                                              columns=['k_val'], aggfunc=np.mean)

                plotting.HeatMapPivot(recall_pivot, title_text=f"Recall with sample size {samples}",
                                      save=True, save_path=path_map+f"recall_samples{samples}.png")
                plotting.HeatMapPivot(coverage_pivot, title_text=f"Coverage with sample size {samples}",
                                      save=True, save_path=path_map+f"coverage_samples{samples}.png")




