import numpy as np
import pandas as pd
from experiments import gaussian_experiment as exp
import matplotlib.pyplot as plt
import visualize as plotting

# TODO Refactoring
def plotHeatMaps(dataframe, map_path, sample_size):
    pr_first_pivot = dataframe.pivot(index="lambda_factor", columns="dimension", values="pr_nearest_distance")
    pr_second_pivot = dataframe.pivot(index="lambda_factor", columns="dimension", values="pr_above_mean")
    dc_first_pivot = dataframe.pivot(index="lambda_factor", columns="dimension", values="dc_nearest_distance")
    dc_second_pivot = dataframe.pivot(index="lambda_factor", columns="dimension", values="dc_above_mean")
    # Precision and Recall
    pr_save_path = f"{map_path}pr_s{sample_size}.png"
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

def runExperiment(distribution_name, sample_sizes, dimensions, lambda_factors, real_scaling, map_path):
    headers = ["dimension", "lambda_factor", "k_val", "pr_above_mean",
               "pr_nearest_distance", "dc_above_mean", "dc_nearest_distance"]
    for sample_size in sample_sizes:
        row_values = []
        k_vals = np.array([1, 3, 7, 9, 16, 32, 64, sample_size - 1])
        for dimension in dimensions:
            reference_distribution, scaled_distributions = exp.getDistributions(distribution_name, sample_size, dimension, lambda_factors)
            for index, scaled_distribution in enumerate(scaled_distributions):
                constant_factor = 1
                scale_factor = lambda_factors[index]
                # Real distribution first argument
                if real_scaling:
                    pr_aboves, dc_aboves, pr_nearest_distances, dc_nearest_distances = exp.doExperiment(distribution_name,
                        scaled_distribution, reference_distribution, scale_factor, constant_factor, k_vals,
                        save_curve=True, map_path=map_path)
                else:
                    pr_aboves, dc_aboves, pr_nearest_distances, dc_nearest_distances = exp.doExperiment(distribution_name,
                        reference_distribution, scaled_distribution,
                        constant_factor, scale_factor, k_vals, save_curve=True, map_path=map_path)

                for index, k_value, in enumerate(k_vals):
                    pr_above = pr_aboves[index].astype(int)
                    pr_near = pr_nearest_distances[index]
                    dc_above = dc_aboves[index].astype(int)
                    dc_near = dc_nearest_distances[index]

                    row = [dimension, scale_factor, k_value, pr_above, pr_near, dc_above, dc_near]
                    row_values.append(row)



        dataframe = pd.DataFrame(data=row_values, columns=headers)
        grouped_data = dataframe.groupby(["dimension", "lambda_factor"]).mean().reset_index()
        plotHeatMaps(grouped_data, map_path, sample_size)

def main():
    # Setup experiment parameters
    sample_sizes = [1000, 3000, 5000]
    sample_sizes = [1000]
    dimensions = [2, 8, 16, 32, 64]
    dimensions = [2]
    lambda_factors = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 1])
    lambda_factors = np.array([0.1, 1, 10])
    distribution_name = "gaussian"
    #distribution_name = "exponential"
    fake_scaled = f"C:/Users/lexme/Documents/gan_thesis_v2/images/{distribution_name}/fake_scaled/"
    runExperiment(distribution_name, sample_sizes, dimensions, lambda_factors, real_scaling=False, map_path=fake_scaled)
    real_scaled = f"C:/Users/lexme/Documents/gan_thesis_v2/images/{distribution_name}/real_scaled/"
    runExperiment(distribution_name, sample_sizes, dimensions, lambda_factors, real_scaling=True, map_path=real_scaled)
    

main()