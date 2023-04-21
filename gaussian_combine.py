import numpy as np
import pandas as pd
from experiments import create_experiment as exp
from experiments import  distributions as dist
import matplotlib.pyplot as plt
import visualize as plotting
import experiments.experiment_visualization as exp_vis
import helper_functions as helper
import helper_functions as util
import save_data as save_data

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
    headers = ["iteration", "dimension", "lambda_factor", "k_val",
                  "pr_nearest_distance", "dc_nearest_distance"]
    row_data = []
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
            pr_distance = pr_rows[i,3]
            dc_distance = dc_rows[i, 3]
            row = other_info + [pr_distance, dc_distance]
            row_data.append(row)

    dataframe = pd.DataFrame(data=row_data, columns=headers)

    return dataframe

def doBoxplots(dataframe, score_names, save_path_map, factors):
    for score_name in score_names:
        exp_vis.saveLambaBoxplotCombine(dataframe,  score_name, save_path_map, factors)

def prepData(factors):
    save_path_distributions = f"./gaussian_combine/data/distributions_{factors}.pkl"
    save_path_distances = f"./gaussian_combine/data/distance_matrices_{factors}.pkl"
    param_dict = {"iterations": 2, "sample_sizes": [1000], "dimensions": [2],
                  "lambda_factors":  factors}
    save_data.saveData(save_path_distributions, save_path_distances, param_dict)

def runGaussian(factors):
    # Read Data
    save_path_distributions = f"./gaussian_combine/data/distributions_{factors}.pkl"
    save_path_distance = f"./gaussian_combine/data/distance_matrices_{factors}.pkl"
    distribution_dict = helper.readPickle(save_path_distributions)
    distance_matrix_dict = helper.readPickle(save_path_distance)
    print("reading done")
    k_vals = [i for i in range(1, 1000, 5)]
    real_scaled_dataframe = runMultiple(distribution_dict, distance_matrix_dict, k_vals, real_scaling=True)
    fake_scaled_dataframe = runMultiple(distribution_dict, distance_matrix_dict, k_vals, real_scaling=False)

    real_map_path = f"./gaussian_combine/real_scaled/"
    fake_map_path = "./gaussian_combine/fake_scaled/"
    score_names = ["pr_nearest_distance", "dc_nearest_distance"]
    doBoxplots(real_scaled_dataframe, score_names, real_map_path, factors)
    doBoxplots(fake_scaled_dataframe, score_names, fake_map_path, factors)
def oneRangeExperiment(save):
    factors = [2**(-i) for i in range(8)]
    print(factors)
    if save:
        prepData(factors)
    runGaussian(factors)
def rangeExperiment(save):
    values = np.arange(0.01, 1.1, .1)
    for index in range(values.shape[0]-1):
        start = values[index]
        end = values[index+1]
        new_factors = np.arange(start, end, 0.05)
        factors_rounded = np.round(new_factors, 2)
        # ! is needed as base
        factors_all = [1] + list(factors_rounded)
        print(factors_all)
        if save:
            prepData(factors_all)
        runGaussian(factors_all)


oneRangeExperiment(save=True)