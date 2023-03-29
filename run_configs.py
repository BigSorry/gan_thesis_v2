import numpy as np
import pandas as pd
from experiments import create_experiment as exp
from experiments import  distributions as dist
import matplotlib.pyplot as plt
import visualize as plotting
import experiments.experiment_visualization as exp_vis

# TODO Refactoring
def plotHeatMaps(dataframe, sample_size, scaling_factor, map_path):
    pr_first_pivot = dataframe.pivot(index="k_val", columns="dimension", values="pr_nearest_distance")
    pr_second_pivot = dataframe.pivot(index="k_val", columns="dimension", values="pr_above_mean")
    dc_first_pivot = dataframe.pivot(index="k_val", columns="dimension", values="dc_nearest_distance")
    dc_second_pivot = dataframe.pivot(index="k_val", columns="dimension", values="dc_above_mean")
    # Precision and Recall
    pr_save_path = f"{map_path}pr_s{sample_size}_{scaling_factor}.png"
    plt.figure(figsize=(14, 6))
    plotting.HeatMapPivot(pr_first_pivot, title_text=f"Precision and Recall with samples {sample_size} \n"
                                                  f"mean l1 distance between pr and nearest theoretical point",
                          save=True, save_path=pr_save_path)

    dc_save_path = f"{map_path}dc_s{sample_size}_{scaling_factor}.png"
    plt.figure(figsize=(14, 6))
    plotting.HeatMapPivot(dc_first_pivot, title_text=f"Density and Coverage with samples {sample_size} \n"
                                                     f"mean l1 distance between pr and nearest theoretical point",
                          save=True, save_path=dc_save_path)

    # Overestimation images
    # plt.figure(figsize=(14, 6))
    # pr_save_path = f"{map_path}pr2_s{sample_size}_{scaling_factor}.png"
    # plotting.HeatMapPivot(pr_second_pivot, title_text=f"percentage points overestimation",
    #                       save=True, save_path=pr_save_path)
    # # Density and Coverage
    # dc_save_path = f"{map_path}dc2_s{sample_size}_{scaling_factor}.png"
    # plt.figure(figsize=(14, 6))
    # plotting.HeatMapPivot(dc_second_pivot, title_text=f"percentage points overestimation",
    #                       save=True, save_path=dc_save_path)

def plotBoxplot(dimensions, scores, save_path):
    plt.figure(figsize=(14, 6))
    exp_vis.boxPlot("", dimensions, scores, save=True, save_path=save_path)

def runExperiment(iteration, distribution_name, k_vals, sample_size, dimension, lambda_factors, real_scaling, map_path):
    rows = []
    save_curve = False
    reference_distribution, scaled_distributions = dist.getDensities(sample_size, dimension, lambda_factors, distribution_name=distribution_name)
    for index, scaled_distribution in enumerate(scaled_distributions):
        constant_factor = lambda_factors[0]
        scale_factor = lambda_factors[index]
        # Real distribution first argument
        if real_scaling:
            pr_aboves, dc_aboves, pr_nearest_distances, dc_nearest_distances = exp.doExperiment(distribution_name,
                scaled_distribution, reference_distribution, scale_factor, constant_factor, k_vals,
                save_curve=save_curve, map_path=map_path)
        else:
            pr_aboves, dc_aboves, pr_nearest_distances, dc_nearest_distances = exp.doExperiment(distribution_name,
                reference_distribution, scaled_distribution,
                constant_factor, scale_factor, k_vals, save_curve=save_curve, map_path=map_path)

        for index, k_value, in enumerate(k_vals):
            pr_above = pr_aboves[index].astype(int)
            pr_near = pr_nearest_distances[index]
            dc_above = dc_aboves[index].astype(int)
            dc_near = dc_nearest_distances[index]

            row = [iteration, dimension, scale_factor, k_value, pr_above, pr_near, dc_above, dc_near]
            rows.append(row)

    return rows

def runMultiple(iterations, distribution_name, k_vals, sample_sizes, dimensions, lambda_factors, real_scaling, map_path):
    headers = ["iteration", "dimension", "lambda_factor", "k_val", "pr_above_mean",
                "pr_nearest_distance", "dc_above_mean", "dc_nearest_distance"]
    all_rows= []
    for iter in range(iterations):
        for samples in sample_sizes:
            k_vals.append(samples-1)
            for dimension in dimensions:
                rows = runExperiment(iter, distribution_name, k_vals, samples, dimension, lambda_factors,
                                 real_scaling=real_scaling, map_path=map_path)
                all_rows.extend(rows)

    dataframe = pd.DataFrame(data=all_rows, columns=headers)
    return dataframe

def plotLambaBoxplot(dataframe, lambda_factors, sample_size, map_path):
    distance_scores = ["pr_nearest_distance" , "dc_nearest_distance"]
    for scale in lambda_factors:
        grouped_data = dataframe.loc[dataframe["lambda_factor"] == scale, :].groupby(["dimension"]) \
            .agg([np.mean, np.std, np.min, np.max]).reset_index()
        dimensions = grouped_data["dimension"]
        for name in distance_scores:
            save_path = f"{map_path}{name[:2]}_lambda_{scale}_s{sample_size}.png"
            distances = grouped_data[name]
            plotBoxplot(dimensions, distances, save_path)
    distance_scores = ["pr_nearest_distance", "dc_nearest_distance"]
    for scale in lambda_factors:
        grouped_data = dataframe.loc[dataframe["lambda_factor"] == scale, :].groupby(["dimension"]) \
            .agg([np.mean, np.std, np.min, np.max]).reset_index()
        dimensions = grouped_data["dimension"]
        for name in distance_scores:
            save_path = f"{map_path}{name[:2]}_lambda_{scale}_s{sample_size}.png"
            distances = grouped_data[name]
            plotBoxplot(dimensions, distances, save_path)

def runGaussian():
    # Key is name which is scaled and value corresponding value for method calling
    scaling_info = {"fake_scaled":False, "real_scaled":True}
    scaling_info = {"fake_scaled":False}
    iterations = 1
    sample_sizes = [2000]
    sample_sizes = [100]
    dimensions = [2, 8, 16, 32, 64]
    dimensions = [2, 8, 16, 32, 64]
    lambda_factors = np.array([1, 0.75, 0.5, 0.25, 0.1, 0.01])
    k_vals = [1, 5, 9]
    distribution_name = "gaussian"
    plot_box = False
    plot_maps = False

    for name, boolean_val in scaling_info.items():
        map_path = f"C:/Users/lexme/Documents/gan_thesis_v2.2/gan_thesis_v2/{distribution_name}/{name}/"

        dataframe = runMultiple(iterations, distribution_name, k_vals, sample_sizes, dimensions,
                                            lambda_factors, real_scaling=boolean_val, map_path=map_path)
        for value in lambda_factors:
            sel_data = dataframe.loc[dataframe["lambda_factor"] == value, :]
            grouped_data = sel_data.groupby(["dimension", "k_val"]).mean().reset_index()
            plotHeatMaps(grouped_data, sample_sizes[0], value, map_path)
        if plot_box:
            map_path = f"C:/Users/lexme/Documents/gan_thesis_v2.2/gan_thesis_v2//{distribution_name}/{name}/lambda/"
            plotLambaBoxplot(dataframe, lambda_factors, sample_sizes[0], map_path)


def runExponential(sample_sizes, dimensions):
    lambda_factors = np.array([0.1, 0.5, 1, 2, 4])
    distribution_name = "exponential"
    fake_scaled = f"C:/Users/lexme/Documents/gan_thesis_v2/images/{distribution_name}/fake_scaled/"

    runExperiment(distribution_name, sample_sizes, dimensions, lambda_factors, real_scaling=False, map_path=fake_scaled)
    real_scaled = f"C:/Users/lexme/Documents/gan_thesis_v2/images/{distribution_name}/real_scaled/"
    runExperiment(distribution_name, sample_sizes, dimensions, lambda_factors, real_scaling=True, map_path=real_scaled)

def runGaussianEqual():
    iterations = 5
    sample_sizes = [5000]
    dimensions = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    k_vals = [1, 3, 5, 7, 9, 32, 64, 128, 256, 512]
    lambda_factors = np.array([1])
    distribution_name = "gaussian"
    map_path = f"./gaussian_equal/"

    real_scaling = True
    dataframe = runMultiple(iterations, distribution_name, k_vals, sample_sizes, dimensions, lambda_factors, real_scaling, map_path)
    grouped_data = dataframe.groupby(["dimension"]).agg([np.mean, np.std, np.min, np.max]).reset_index()
    plotBoxplot(grouped_data, map_path, sample_sizes[0])

def main():
    # Setup experiment parameters
    sample_sizes = [1000, 3000, 5000]
    sample_sizes = [1000]
    dimensions = [2, 8, 16, 32, 64]
    dimensions = [2]

    #runGaussianEqual()
    gaussianLine()
    #runExponential(sample_sizes, dimensions)


main()