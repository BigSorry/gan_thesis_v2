import numpy as np
import pandas as pd
import union_experiment as exp
import matplotlib.pyplot as plt
import visualize as plotting

# TODO Refactoring
def plotHeatMaps(dataframe, map_path, sample_size):
    pr_first_pivot = dataframe.pivot(index="lambda_factor", columns="dimension", values="pr_nearest_distances")
    pr_second_pivot = dataframe.pivot(index="lambda_factor", columns="dimension", values="pr_under_mean")
    dc_first_pivot = dataframe.pivot(index="lambda_factor", columns="dimension", values="dc_nearest_distances")
    dc_second_pivot = dataframe.pivot(index="lambda_factor", columns="dimension", values="dc_under_mean")
    # Precision and Recall
    pr_save_path = f"{map_path}pr_s{sample_size}_.png"
    plt.figure(figsize=(14, 6))
    plt.subplot(2, 1, 1)
    plotting.HeatMapPivot(pr_first_pivot, title_text=f"Precision and Recall with samples {sample_size} \n"
                                                  f"mean l1 distance between pr and nearest theoretical point",
                          save=False, save_path=pr_save_path)
    plt.subplot(2, 1, 2)
    plotting.HeatMapPivot(pr_second_pivot, title_text=f"percentage points overestimation",
                          save=True, save_path=pr_save_path)
    # Density and Coverage
    dc_save_path = f"{map_path}dc_s{sample_size}.png"
    plt.figure(figsize=(14, 6))
    plt.subplot(2, 1, 1)
    plotting.HeatMapPivot(dc_first_pivot, title_text=f"Density and Coverage with samples {sample_size} \n"
                                                     f"mean l1 distance between pr and nearest theoretical point",
                          save=False, save_path=dc_save_path)
    plt.subplot(2, 1, 2)
    plotting.HeatMapPivot(dc_second_pivot, title_text=f"percentage points overestimation",
                          save=True, save_path=dc_save_path)

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
               "pr_nearest_distances", "dc_nearest_distances", "real_scaling"]
    sample_sizes = [1000, 3000, 5000]
    #sample_sizes = [1000]
    dimensions = [2, 8, 16, 32, 64]
    lambda_factors = np.array([0.01, 0.25, 0.5, 0.75, 1])
    for sample_size in sample_sizes:
        row_values = []
        k_vals = np.array([1, 2, 3, 4, 8, 9, 16, 32, 64, sample_size - 1])
        for dimension in dimensions:
            reference_distribution, scaled_distributions = exp.getDistributions(sample_size, dimension, lambda_factors)
            for index, distribution in enumerate(scaled_distributions):
                constant_factor = 1
                scale_factor = lambda_factors[index]
                stat_values = exp.doExperiment(reference_distribution, distribution, constant_factor, scale_factor,
                                 k_vals, save_curve=False)
                stat_values_reverse = exp.doExperiment(distribution, reference_distribution, scale_factor, constant_factor,
                                               k_vals, save_curve=False)

                row = [dimension, scale_factor] + stat_values + [True]
                row_reverse = [dimension, scale_factor] + stat_values_reverse + [False]
                row_values.append(row)
                row_values.append(row_reverse)

        dataframe = pd.DataFrame(data=row_values, columns=headers)
        real_scaled = dataframe.loc[dataframe["real_scaling"] == True, :]
        fake_scaled_scaled = dataframe.loc[dataframe["real_scaling"] == False, :]
        fake_map_path = "C:/Users/lexme/Documents/gan_thesis_v2/images/fake_scaled/"
        real_map_path = "C:/Users/lexme/Documents/gan_thesis_v2/images/real_scaled/"

        plotHeatMaps(real_scaled, fake_map_path, sample_size)
        plotHeatMaps(fake_scaled_scaled, real_map_path, sample_size)


runExperiment()