import numpy as np
import matplotlib.pyplot as plt
import experiments.experiment_visualization as exp_vis
import check_densities as ch_den
from experiments import experiment_calc as exp
import helper_functions as util

def doCalcs(sample_size, dimension, ratios, real_scaling=False):
    k_vals = [i for i in range(1, sample_size)]
    base_value = 1
    pr_calc = np.zeros((len(k_vals), len(ratios)))
    dc_calc = np.zeros((len(k_vals), len(ratios)))
    for index, ratio in enumerate(ratios[1:]):
        scale = base_value*ratio
        lambda_factors = [base_value, scale]
        reference_distribution, scaled_distribution = ch_den.getGaussian(sample_size, dimension, lambda_factors)
        if real_scaling:
            lambda_factors = [scale, base_value]
            distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = util.getDistanceMatrices(scaled_distribution, reference_distribution)
            curve_classifier = exp.getCurveClassifier("gaussian", scaled_distribution, reference_distribution, lambda_factors)
        else:
            distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = util.getDistanceMatrices(reference_distribution, scaled_distribution)
            curve_classifier = exp.getCurveClassifier("gaussian", reference_distribution, scaled_distribution, lambda_factors)

        pr_pairs, dc_pairs = exp.getKNN(distance_matrix_real, distance_matrix_fake, distance_matrix_pairs, k_vals)
        pr_aboves, pr_nearest_distances = exp.getStats(curve_classifier, pr_pairs)
        dc_aboves, dc_nearest_distances = exp.getStats(curve_classifier, dc_pairs)

        pr_calc[:, index] = pr_nearest_distances
        dc_calc[:, index] = dc_nearest_distances

    return pr_calc, dc_calc

def saveBoxplot(score_results, metric_name, map_path):
    k_vals = np.arange(score_results.shape[0]) + 1
    means = np.mean(score_results, axis=1)
    stds = np.std(score_results, axis=1)

    plt.figure(figsize=(14, 6))
    plt.errorbar(k_vals, means, stds, linestyle='None', marker='o')
    plt.ylim([0, 1.1])
    plt.xlabel("K-value")
    save_path = f"{map_path}/{metric_name}.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()