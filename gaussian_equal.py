import os
import pandas as pd
import numpy as np
from experiments import create_experiment as exp
from experiments import  distributions as dist
import matplotlib.pyplot as plt
import visualize as plotting
import experiments.experiment_visualization as exp_vis
import helper_functions as helper

# TODO Refactoring
def plotCurve(pr_dataframe, dc_dataframe, th_curve_dict, map_path):
    for key, curve in th_curve_dict.items():
        plt.figure()
        dimension = key[2]
        other_factor = key[3]
        scale_factors = [1, other_factor]
        select_pr = pr_dataframe.loc[(pr_dataframe["dimension"] == dimension), :]
        select_dc = dc_dataframe.loc[(dc_dataframe["dimension"] == dimension), :]
        pr_pairs = select_pr.loc[:, ["precision", "recall"]].values
        dc_pairs = select_dc.loc[:, ["density", "coverage"]].values
        k_vals = select_pr["k_val"].unique()
        dimension_map = f"{map_path}/curve/scale={other_factor}_d={dimension}/"
        if not os.path.exists(dimension_map):
            os.makedirs(dimension_map)
        save_path = f"{dimension_map}/params_{other_factor}.png"
        # if dimension == 2:
        #     plt.subplot(1, 2, 1)
        #     exp_vis.plotTheoreticalCurve(curve, curve, scale_factors, save=False)
        #     exp_vis.plotKNNMetrics(pr_pairs, dc_pairs, k_vals, save_path, save=False)
        #     plt.subplot(1, 2, 2)
        #     exp_vis.plotDistributions(distribution, other_distribution, "", save_path, save=True)
        # else:
        exp_vis.plotTheoreticalCurve(curve, curve, scale_factors, save=False)
        exp_vis.plotKNNMetrics(pr_pairs, dc_pairs, k_vals, save_path, save=True)

def runExperiment(reference_distribution, scaled_distribution , k_vals, lambda_params, real_scaling):
    constant_factor = lambda_params[0]
    scale_factor = lambda_params[1]
    # Real distribution first argument
    if real_scaling:
        pr_pairs, dc_pairs = exp.getKNN(scaled_distribution, reference_distribution, k_vals)
        curve_classifier, curve_var_dist = exp.getGroundTruth("gaussian", scaled_distribution, reference_distribution, [scale_factor, constant_factor])
    else:
        pr_pairs, dc_pairs = exp.getKNN(reference_distribution, scaled_distribution, k_vals)
        curve_classifier, curve_var_dist = exp.getGroundTruth("gaussian", reference_distribution, scaled_distribution, [constant_factor, scale_factor])

    pr_aboves, pr_nearest_distances = exp.getStats(curve_var_dist, pr_pairs)
    dc_aboves, dc_nearest_distances = exp.getStats(curve_var_dist, dc_pairs)

    pr_rows = np.array([pr_pairs[:, 0], pr_pairs[:, 1], pr_aboves.astype(int),  pr_nearest_distances]).T
    dc_rows = np.array([dc_pairs[:, 0], dc_pairs[:, 1], dc_aboves.astype(int),  dc_nearest_distances]).T

    return pr_rows, dc_rows, curve_var_dist

def runMultiple(distribution_dict, k_vals, real_scaling):
    pr_headers = ["iteration", "dimension", "lambda_factor", "k_val",
                  "precision", "recall", "pr_above", "pr_nearest_distance"]
    dc_headers = ["iteration", "dimension", "lambda_factor", "k_val",
                  "density", "coverage", "dc_above", "dc_nearest_distance"]
    pr_data = []
    dc_data = []
    th_curve_dict = {}
    standard_scale = 1
    for key, other_distribution in distribution_dict.items():
        (iter, sample_size, dimension, other_scale) = key
        reference_distribution = distribution_dict[(iter, sample_size, dimension, standard_scale)]
        lambda_params = [standard_scale, other_scale]
        pr_rows, dc_rows, th_curve = runExperiment(reference_distribution, other_distribution, k_vals, lambda_params, real_scaling=real_scaling)
        th_curve_dict[(iter, sample_size, dimension, other_scale)] = th_curve
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

    return pr_dataframe, dc_dataframe, th_curve_dict

def runGaussianEqual():
    iterations = 10
    sample_sizes = [1000]
    dimensions = [2, 512]
    k_vals = [i for i in range(1, sample_sizes[0], 5)]
    lambda_factors = np.array([1])
    distribution_name = "gaussian"
    map_path = f"./gaussian_equal/"
    save_path_distributions = "./gaussian_equal/data/distributions.pkl"
    save_distributions = True

    if save_distributions:
        distribution_dict = dist.saveDistributions(iterations, sample_sizes, dimensions, lambda_factors, save_path_distributions)
    else:
        distribution_dict = helper.readPickle(save_path_distributions)

    real_scaling = True
    pr_dataframe, dc_dataframe, th_curve_dict = runMultiple(distribution_dict, k_vals, real_scaling)
    score_names = ["pr_nearest_distance"]
    exp_vis.plotLambaBoxplot(pr_dataframe, dimensions, lambda_factors, score_names, map_path)
    score_names = ["dc_nearest_distance"]
    exp_vis.plotLambaBoxplot(dc_dataframe, dimensions, lambda_factors, score_names, map_path)



    # pr_grouped = pr_dataframe.groupby(["dimension", "k_val" ])["precision", "recall"].agg([np.mean]).reset_index()
    # dc_grouped = dc_dataframe.groupby(["dimension", "k_val" ])["precision", "recall"].agg([np.mean]).reset_index()
    # plotCurve(pr_grouped, dc_grouped, th_curve_dict, map_path)



runGaussianEqual()