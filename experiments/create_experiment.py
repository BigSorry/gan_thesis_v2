import os
import numpy as np
import matplotlib.pyplot as plt
import metrics.likelihood_classifier as llc
import experiments.likelihood_estimations as ll_est
import experiments.plotting as exp_plot
import helper_functions as util
import metrics.not_used.metrics as mtr
from sklearn.neighbors import KNeighborsClassifier

def getGroundTruth(distribution_name, real_data, fake_data, scale_factors):
    lambdas = llc.getPRLambdas(angle_count=1000)
    epsilon = 10e-16
    densities_real, densities_fake = ll_est.getDensities(real_data, fake_data, scale_factors, method_name=distribution_name)
    densities_real_norm = densities_real / (np.sum(densities_real) + epsilon)
    densities_fake_norm = densities_fake / (np.sum(densities_fake) + epsilon)

    curve_class = []
    curve_distance = []
    for value in lambdas:
        predictions = (value*densities_real >= densities_fake).astype(int)
        precision_hist = value*1 - np.sum(np.abs(value*densities_real_norm - densities_fake_norm))
        recall_hist = (1/value)*1 - np.sum(np.abs(densities_real_norm - densities_fake_norm*(1/value)))
        truth_labels = np.concatenate([np.ones(real_data.shape[0]), np.zeros(fake_data.shape[0])])
        fpr, fnr = llc.getScores(truth_labels, predictions)
        precision_class = value*fpr + fnr
        recall_class = precision_class / value
        curve_class.append([np.clip(precision_class,0,1), np.clip(recall_class,0,1)])
        curve_distance.append([np.clip(precision_hist,0,1), np.clip(recall_hist,0,1)])

    return np.array(curve_class), np.array(curve_distance)

def getKNN(real_data, fake_data, k_vals):
    distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = util.getDistanceMatrices(real_data, fake_data)
    pr_pairs = np.zeros((k_vals.shape[0], 2))
    dc_pairs = np.zeros((k_vals.shape[0], 2))
    for i, k_val in enumerate(k_vals):
        boundaries_real = distance_matrix_real[:, k_val]
        boundaries_fake = distance_matrix_fake[:, k_val]
        precision, recall, density, coverage = util.getScores(distance_matrix_pairs, boundaries_fake, boundaries_real, k_val)
        pr_pairs[i, :] = [precision, recall]
        dc_pairs[i, :] = [density, coverage]

    return [pr_pairs, dc_pairs]

## Not used atm
def getCurves(real_data, fake_data):
    params = {"k_cluster": 20, "angles": 1001, "kmeans_runs": 2}
    pr_score, curve, cluster_labels = mtr.getHistoPR(real_data, fake_data, params)
    classifier = KNeighborsClassifier(n_neighbors=1)
    params = {"threshold_count": 100, "angles": 1001, "classifier": classifier}
    pr_score, curve2, prob_labels = mtr.getClassifierPR(real_data, fake_data, params)

    return [curve, curve2]

def getStats(curve, metric_points):
    points_above = []
    nearest_distance = []
    for point in metric_points:
        # Clipping for density score
        point = np.clip(point, 0, 1)
        distances = point - curve
        l1_distances = np.sum(np.abs(distances), axis=1)
        l1_distances_sorted = np.sort(l1_distances)
        nearest_l1_distance  = l1_distances_sorted[: 1].mean()
        nearest_distance.append(nearest_l1_distance)

        row_check = (distances[:, 0] >= 0) & (distances[:, 1] >= 0)
        above_true = np.sum(row_check) > 0
        points_above.append(above_true)

    return np.array(points_above), np.array(nearest_distance)

def doExperiment(distribution_name, distribution, other_distribution, real_factor, other_factor, k_vals, save_curve=False, map_path=""):
    curve_methods = False
    pr_pairs, dc_pairs = getKNN(distribution, other_distribution, k_vals)
    scale_factors = [real_factor, other_factor]
    curve_classifier, curve_var_dist = getGroundTruth(distribution_name, distribution, other_distribution, scale_factors)
    pr_above, pr_nearest_distances = getStats(curve_var_dist, pr_pairs)
    dc_above, dc_nearest_distances = getStats(curve_var_dist, dc_pairs)

    if save_curve:
        plt.figure()
        dimension = distribution.shape[1]
        sample_size = distribution.shape[0]
        dimension_map = f"{map_path}/curve/s={sample_size}_d={dimension}/"
        if not os.path.exists(dimension_map):
            os.makedirs(dimension_map)
        save_path = f"{dimension_map}/params_r{real_factor}_f{other_factor}.png"
        if dimension == 2:
            plt.subplot(1, 2, 1)
            exp_plot.plotTheoreticalCurve(curve_classifier, curve_var_dist, scale_factors, save=False)
            exp_plot.plotKNNMetrics(pr_pairs, dc_pairs, pr_above, dc_above, k_vals, save_path, save=False)
            plt.subplot(1, 2, 2)
            exp_plot.plotDistributions(distribution, other_distribution, "", save_path, save=True)
        else:
            exp_plot.plotTheoreticalCurve(curve_classifier, curve_var_dist, scale_factors, save=False)
            exp_plot.plotKNNMetrics(pr_pairs, dc_pairs, pr_above, dc_above, k_vals, save_path, save=True)

        # plt.subplot(1, 3, 3)
        # plotStats(pr_above, dc_above, save_path, save=save_curve)

    return pr_above, dc_above, pr_nearest_distances, dc_nearest_distances




