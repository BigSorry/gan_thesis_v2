import os
import numpy as np
import matplotlib.pyplot as plt
import metrics.likelihood_classifier as llc
import experiments.likelihood_estimations as ll_est
import experiments.experiment_visualization as exp_plot
import helper_functions as util
import metrics.not_used.metrics as mtr
from sklearn.neighbors import KNeighborsClassifier

# Theoretical with likelihood
def getCurveVarDistance(distribution_name, real_data, fake_data, scale_factors):
    lambdas = llc.getPRLambdas(angle_count=1000)
    densities_real, densities_fake = ll_est.getDensities(real_data, fake_data, scale_factors,
                                                         method_name=distribution_name, norm=True)
    curve_pairs = []
    for scale in lambdas:
        density_real_scale = densities_real*scale
        density_fake_scale = densities_fake*(1/scale)
        densities_real_norm = density_real_scale / (np.sum(density_real_scale) )
        densities_fake_norm = density_fake_scale / (np.sum(density_fake_scale) )
        diff_prec = scale - np.sum(np.abs(densities_real_norm - densities_fake))
        diif_recall = (1/scale) - np.sum(np.abs(densities_fake_norm - densities_real))
        curve_pairs.append([np.clip(diff_prec,0,1), np.clip(diif_recall,0,1)])

    return np.array(curve_pairs)

# Theoretical with likelihood
def getCurveClassifier(distribution_name, real_data, fake_data, scale_factors):
    lambdas = llc.getPRLambdas(angle_count=1000)
    densities_real, densities_fake = ll_est.getDensities(real_data, fake_data, scale_factors,
                                                         method_name=distribution_name, norm=False)
    curve_pairs = []
    for scale in lambdas:
        predictions = (densities_real*scale >= densities_fake).astype(int)
        truth_labels = np.concatenate([np.ones(real_data.shape[0]), np.zeros(fake_data.shape[0])])
        fpr, fnr = llc.getScores(truth_labels, predictions)
        precision_class = scale*fpr + fnr
        recall_class = precision_class / scale
        curve_pairs.append([np.clip(precision_class,0,1), np.clip(recall_class,0,1)])

    return np.array(curve_pairs)
def getKNNData(real_data, fake_data, k_vals):
    distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = util.getDistanceMatrices(real_data, fake_data)
    pr_pairs = np.zeros((len(k_vals), 2))
    dc_pairs = np.zeros((len(k_vals), 2))
    for i, k_val in enumerate(k_vals):
        boundaries_real = distance_matrix_real[:, k_val]
        boundaries_fake = distance_matrix_fake[:, k_val]
        precision, recall, density, coverage = util.getScores(distance_matrix_pairs, boundaries_fake, boundaries_real, k_val)
        pr_pairs[i, :] = [precision, recall]
        dc_pairs[i, :] = [density, coverage]

    return [pr_pairs, dc_pairs]

def getKNN(distance_real, distance_fake, distance_pairs, k_vals):
    pr_pairs = np.zeros((len(k_vals), 2))
    dc_pairs = np.zeros((len(k_vals), 2))
    for i, k_val in enumerate(k_vals):
        boundaries_real = distance_real[:, k_val]
        boundaries_fake = distance_fake[:, k_val]
        precision, recall, density, coverage = util.getScores(distance_pairs, boundaries_fake, boundaries_real, k_val)
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






