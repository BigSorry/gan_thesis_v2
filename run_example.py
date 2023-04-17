import numpy as np
import pandas as pd
from experiments import create_experiment as exp
from experiments import  distributions as dist
import matplotlib.pyplot as plt
import visualize as plotting
import experiments.experiment_visualization as exp_vis
import helper_functions as helper
import helper_functions as util

def doPlots(curve_var_dist, reference_distribution, scaled_distribution, lambda_factors,
            pr_pairs, dc_pairs, k_vals, save_path):
    real_z = -1
    fake_z = 1
    if lambda_factors[0] < lambda_factors[1]:
        real_z = 1
        fake_z = -1
    plt.subplot(1, 2, 1)
    exp_vis.plotTheoreticalCurve(curve_var_dist, curve_var_dist, lambda_factors, save=False)
    exp_vis.plotKNNMetrics(pr_pairs, dc_pairs, k_vals, save_path, save=False)
    plt.subplot(1, 2, 2)
    exp_vis.plotDistributions(reference_distribution, scaled_distribution, real_z, fake_z, "", save_path, save=True)

def getK(sample_size, low_boundary=10, step_low=2, step_high=50):
    low_k = [i for i in range(1, low_boundary, step_low)]
    high_k = [50, 100, 500, 750, 999]
    all_k = low_k + high_k

    return all_k

def runExample(real_scaled):
    save_path_distributions = "./gaussian/data/distributions.pkl"
    save_path_distance = "./gaussian/data/distance_matrices.pkl"
    distribution_dict = helper.readPickle(save_path_distributions)
    distance_matrix_dict = helper.readPickle(save_path_distance)
    sample_size = 1000
    k_vals = getK(sample_size, low_boundary=10, step_low=3, step_high=400)
    constant_factor = 1
    scale_factors = [0.1, 0.25, 0.5, 0.75, 1]
    real_save_path = "./gaussian/curves/real_scaled/"
    fake_save_path = "./gaussian/curves/fake_scaled/"
    reference_key = (0, sample_size, 2, constant_factor)
    reference_distribution = distribution_dict[reference_key]

    for scale_factor in scale_factors:
        scaled_key = (0, sample_size, 2, scale_factor)
        distance_dict = distance_matrix_dict[reference_key][scaled_key]
        scaled_distribution = distribution_dict[scaled_key]
        lambda_factors = [constant_factor, scale_factor]
        if real_scaled:
            lambda_factors = [scale_factor, constant_factor]
            pr_pairs, dc_pairs = exp.getKNN(distance_dict["real"], distance_dict["fake"], distance_dict["real_fake"],
                                            k_vals)
            curve_classifier, curve_var_dist = exp.getGroundTruth("gaussian", scaled_distribution,
                                                                  reference_distribution,
                                                                  [scale_factor, constant_factor])
            doPlots(curve_var_dist, scaled_distribution, reference_distribution, lambda_factors,
                    pr_pairs, dc_pairs, k_vals, real_save_path+f"{scale_factor}.png")
        else:
            pr_pairs, dc_pairs = exp.getKNN(distance_dict["fake"], distance_dict["real"], distance_dict["real_fake"],
                                            k_vals)
            curve_classifier, curve_var_dist = exp.getGroundTruth("gaussian", reference_distribution,
                                                                  scaled_distribution, lambda_factors)
            doPlots(curve_var_dist, reference_distribution, scaled_distribution, lambda_factors,
                    pr_pairs, dc_pairs, k_vals, fake_save_path+f"{scale_factor}.png")


runExample(real_scaled=True)
runExample(real_scaled=False)
