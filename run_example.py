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
    exp_vis.plotKNNMetrics(pr_pairs,  k_vals, "PR_KNN", "black", save_path, save=False)
    exp_vis.plotKNNMetrics(dc_pairs,  k_vals, "DC_KNN", "yellow", save_path, save=False)
    plt.legend(loc='upper center', bbox_to_anchor=(0, 1.15),
               fancybox=True, shadow=True, ncol=2, fontsize=9)
    plt.subplot(1, 2, 2)
    exp_vis.plotDistributions(reference_distribution, scaled_distribution, real_z, fake_z, "", save_path, save=True)

def singlePlot(curve_var_dist, lambda_factors, pr_pairs, dc_pairs, k_vals, save_path):
    plt.figure()
    exp_vis.plotTheoreticalCurve(curve_var_dist, curve_var_dist, lambda_factors, save=False)
    exp_vis.plotKNNMetrics(pr_pairs,  k_vals, "PR_KNN", "black", save_path, save=False)
    exp_vis.plotKNNMetrics(dc_pairs,  k_vals, "DC_KNN", "yellow", save_path, save=True)


def getK(sample_size, low_boundary=10, step_low=2, step_high=50):
    low_k = [i for i in range(1, low_boundary, step_low)]
    high_k = [i for i in range(low_boundary, sample_size, step_high)]
    #high_k = [50, 100, 500, 750, 999]
    all_k = low_k + high_k

    return all_k

def runExample(factors, dimensions, save_path_distributions, save_path_distance, real_scaled):
    distribution_dict = helper.readPickle(save_path_distributions)
    distance_matrix_dict = helper.readPickle(save_path_distance)
    sample_size = 1000
    k_vals = getK(sample_size, low_boundary=10, step_low=1, step_high=10)
    base_scale  = (list(distance_matrix_dict.keys())[0])[-1]
    real_save_path = f"./gaussian_dimension/curves/real_scaled/"
    fake_save_path = f"./gaussian_dimension/curves/fake_scaled/"

    for scale_factor in factors:
        for dim in dimensions:
            save_name = f"dim{dim}_scale{scale_factor}.png"
            reference_key = (0, sample_size, dim, base_scale)
            reference_distribution = distribution_dict[reference_key]
            scaled_key = (0, sample_size, dim, scale_factor)
            distance_dict = distance_matrix_dict[reference_key][scaled_key]
            scaled_distribution = distribution_dict[scaled_key]
            lambda_factors = [base_scale, scale_factor]
            if real_scaled:
                save_path = real_save_path + save_name
                lambda_factors = [scale_factor, base_scale]
                pr_pairs, dc_pairs = exp.getKNN(distance_dict["real"], distance_dict["fake"], distance_dict["real_fake"], k_vals)
                curve_classifier, curve_var_dist = exp.getGroundTruth("gaussian", scaled_distribution, reference_distribution, lambda_factors)
            else:
                save_path = fake_save_path + save_name
                pr_pairs, dc_pairs = exp.getKNN(distance_dict["fake"], distance_dict["real"], distance_dict["real_fake"],  k_vals)
                curve_classifier, curve_var_dist = exp.getGroundTruth("gaussian", reference_distribution,  scaled_distribution, lambda_factors)

            if dim == 2:
                doPlots(curve_var_dist, reference_distribution, scaled_distribution, lambda_factors,
                        pr_pairs, dc_pairs, k_vals, save_path)
            else:
                singlePlot(curve_var_dist, lambda_factors, pr_pairs, dc_pairs, k_vals, save_path)


dimensions = [64]
factors = [.2*1.1 ** (-i) for i in range(8)]
factors = np.round(factors, 4)
print(factors)
save_path_distributions = f"./gaussian_dimension/data/distributions_{factors}.pkl"
save_path_distance = f"./gaussian_dimension/data/distance_matrices_{factors}.pkl"
runExample(factors, dimensions, save_path_distributions, save_path_distance, real_scaled=True)
runExample(factors, dimensions, save_path_distributions, save_path_distance, real_scaled=False)
