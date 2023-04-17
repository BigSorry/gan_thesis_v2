import numpy as np
import pandas as pd
from experiments import create_experiment as exp
from experiments import  distributions as dist
import matplotlib.pyplot as plt
import visualize as plotting
import experiments.experiment_visualization as exp_vis
import helper_functions as helper
import save_data as save_data

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

def runExperiment(distance_dict, reference_distribution, scaled_distribution , k_vals, lambda_params, real_scaling):
    constant_factor = lambda_params[0]
    scale_factor = lambda_params[1]
    # Real distribution first argument
    if real_scaling:
        pr_pairs, dc_pairs = exp.getKNN(distance_dict["fake"], distance_dict["real"], distance_dict["real_fake"], k_vals)
        curve_classifier, curve_var_dist = exp.getGroundTruth("gaussian", scaled_distribution, reference_distribution, [scale_factor, constant_factor])
    else:
        pr_pairs, dc_pairs = exp.getKNN(distance_dict["real"], distance_dict["fake"], distance_dict["real_fake"], k_vals)
        curve_classifier, curve_var_dist = exp.getGroundTruth("gaussian", reference_distribution, scaled_distribution, [constant_factor, scale_factor])

    pr_aboves, pr_nearest_distances = exp.getStats(curve_var_dist, pr_pairs)
    dc_aboves, dc_nearest_distances = exp.getStats(curve_var_dist, dc_pairs)

    pr_rows = np.array([pr_pairs[:, 0], pr_pairs[:, 1], pr_aboves.astype(int),  pr_nearest_distances]).T
    dc_rows = np.array([dc_pairs[:, 0], dc_pairs[:, 1], dc_aboves.astype(int),  dc_nearest_distances]).T

    return pr_rows, dc_rows

def runMultiple(distribution_dict, distance_matrix_dict, k_vals, real_scaling):
    pr_headers = ["iteration", "dimension", "lambda_factor", "k_val",
                  "precision", "recall", "pr_above", "pr_nearest_distance"]
    dc_headers = ["iteration", "dimension", "lambda_factor", "k_val",
                  "density", "coverage", "dc_above", "dc_nearest_distance"]
    pr_data = []
    dc_data = []
    standard_scale = 1
    for key, other_distribution in distribution_dict.items():
        (iter, sample_size, dimension, other_scale) = key
        lambda_params = [standard_scale, other_scale]
        reference_key = (iter, sample_size, dimension, standard_scale)
        reference_distribution = distribution_dict[reference_key]
        distance_dict = distance_matrix_dict[reference_key][key]
        pr_rows, dc_rows = runExperiment(distance_dict, reference_distribution, other_distribution, k_vals, lambda_params, real_scaling=real_scaling)
        for i in range(pr_rows.shape[0]):
            other_info = [iter, dimension, other_scale,  k_vals[i]]
            pr_results = list(pr_rows[i])
            dc_results = list(dc_rows[i])
            pr_row = other_info + pr_results
            dc_row = other_info + dc_results
            pr_data.append(pr_row)
            dc_data.append(dc_row)

    pr_dataframe = pd.DataFrame(data=pr_data, columns=pr_headers)
    dc_dataframe = pd.DataFrame(data=dc_data, columns=dc_headers)

    return pr_dataframe, dc_dataframe

def doBoxplots(dataframe, score_names, save_path_map):
    for score_name in score_names:
        exp_vis.saveLambaBoxplot(dataframe,  score_name, save_path_map)

def prepData():
    save_path_distributions = "./gaussian/data/distributions.pkl"
    save_path_distances = "./gaussian/data/distance_matrices.pkl"
    param_dict = {"iterations": 2, "sample_sizes": [1000], "dimensions": [2],
                  "lambda_factors":  np.array([1, 0.75, 0.5, 0.25, 0.1, 0.01])}
    save_data.saveData(save_path_distributions, save_path_distances, param_dict)

def runGaussian():
    t = np.array([1, 0.75, 0.5, 0.25, 0.1, 0.01])
    # Read Data
    save_path_distributions = "./gaussian/data/distributions.pkl"
    save_path_distance = "./gaussian/data/distance_matrices.pkl"
    distribution_dict = helper.readPickle(save_path_distributions)
    distance_matrix_dict = helper.readPickle(save_path_distance)
    print("reading done")
    k_vals = [i for i in range(1, 1000, 1)]
    pr_dataframe, dc_dataframe = runMultiple(distribution_dict, distance_matrix_dict, k_vals, real_scaling=True)
    fake_pr_dataframe, fake_dc_datafram = runMultiple(distribution_dict, distance_matrix_dict, k_vals, real_scaling=False)

    real_map_path = "./gaussian/real_scaled/"
    fake_map_path = "./gaussian/fake_scaled/"
    score_names = ["pr_nearest_distance"]
    doBoxplots(pr_dataframe, score_names, real_map_path)
    doBoxplots(fake_pr_dataframe, score_names, fake_map_path)

    score_names = ["dc_nearest_distance"]
    doBoxplots(dc_dataframe, score_names, real_map_path)
    doBoxplots(fake_dc_datafram, score_names, fake_map_path)




#prepData()
runGaussian()

