import numpy as np
import pandas as pd
import union_experiment as exp
import matplotlib.pyplot as plt
import visualize as plotting

def plotHeatMaps(first_pivot, second_pivot, title_text, save_path):
    plt.figure(figsize=(14, 6))
    plt.subplot(2, 1, 1)
    plotting.HeatMapPivot(first_pivot, title_text=f"{title_text} "
                                                  f"mean l1 distance between pr and nearest theoretical point",
                          save=False, save_path=save_path)
    plt.subplot(2, 1, 2)
    plotting.HeatMapPivot(second_pivot, title_text=f"{title_text}"
                                                   f"percentage points overestimation",
                          save=True, save_path=save_path)
def plotHeatmaps(dataframe, sample_size):
    # Precision and Recall heatmaps
    map_path = "C:/Users/lexme/Documents/gan_thesis_v2/images/"

    pr_pivot = dataframe.pivot(index="lambda_factor", columns="dimension",
                                      values="pr_nearest_distances")
    pr_pivot2 = dataframe.pivot(index="lambda_factor", columns="dimension", values="pr_under_mean")
    title_text = f"Precision and Recall, samples {sample_size}, \n"
    save_path = f"{map_path}pr_s{sample_size}.png"
    plotHeatMaps(pr_pivot, pr_pivot2, title_text, save_path)

    # Density and Coverage heatmaps
    dc_pivot = dataframe.pivot(index="lambda_factor", columns="dimension",
                                      values="dc_nearest_distances")
    dc_pivot2 = dataframe.pivot(index="lambda_factor", columns="dimension", values="dc_under_mean")
    title_text = f"Density and Coverage, samples {sample_size}, \n"
    save_path = f"{map_path}dc_s{sample_size}.png"
    plotHeatMaps(dc_pivot,dc_pivot2, title_text, save_path)

def checkScaling():
    samples=1000
    dimension = 2
    runs = 3
    var_factors = np.array([0.01, 0.1, 0.2, 0.25, 0.5, 0.75, 1]) / 2

    for i in range(runs):
        exp.doExperiment(sample_size=samples, dimension=dimension, lambda_factors=var_factors)
        var_factors *= 10

def runExperiment():
    headers = ["dimension", "lambda_factor", "pr_under_mean", "dc_under_mean",
               "pr_nearest_distances", "dc_nearest_distances"]
    row_values = []
    sample_sizes = [1000, 3000, 5000]
    sample_sizes = [1000]
    dimensions = [2, 8, 16, 32]
    lambda_factors = np.array([0.01, 0.1, 0.2, 0.25, 0.5, 0.75, 1])
    for sample_size in sample_sizes:
        k_vals = np.array([1, 2, 3, 4, 8, 9, 16, 32, 64, sample_size - 1])
        for dimension in dimensions:
            reference_distribution, scaled_distributions = exp.getDistributions(sample_size, dimension, lambda_factors)
            for index, distribution in enumerate(scaled_distributions):
                constant_factor = 1
                scale_factor = lambda_factors[index]
                stat_values = exp.doExperiment(reference_distribution, distribution, constant_factor, scale_factor,
                                 k_vals, save_curve=False)
                row = [dimension, scale_factor] + stat_values
                row_values.append(row)

        dataframe = pd.DataFrame(data=row_values, columns=headers)
        plotHeatmaps(dataframe, sample_size)

runExperiment()