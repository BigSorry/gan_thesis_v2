import numpy as np
import metrics.likelihood_classifier as llc
import experiments.likelihood_estimations as ll_est
from utility_scripts import helper_functions as util
import metrics.not_used.metrics as mtr
from sklearn.neighbors import KNeighborsClassifier


def getHist(real_density, fake_density):
    lambdas = llc.getPRLambdas(angle_count=1000)
    slopes_2d = np.expand_dims(lambdas, 1)
    ref_dist_2d = np.expand_dims(real_density, 0)
    eval_dist_2d = np.expand_dims(fake_density, 0)

    # Compute precision and recall for all angles in one step via broadcasting
    precision = np.minimum(ref_dist_2d * slopes_2d, eval_dist_2d).sum(axis=1)
    recall = precision / lambdas
    precision = np.clip(precision, 0, 1)
    recall = np.clip(recall, 0, 1)
    return precision, recall
# Theoretical with likelihood
def getCurveVarDistance(distribution_name, real_data, fake_data, scale_factors):
    lambdas = llc.getPRLambdas(angle_count=1000)
    densities_real, densities_fake = ll_est.getDensities(real_data, fake_data, scale_factors,
                                                         method_name=distribution_name, norm=False)
    curve_pairs = []
    params = {"k_cluster":20, "angles":1001, "kmeans_runs":1}
    pr_score, curve, cluster_labels = mtr.getHistoPR(real_data, fake_data, params)
    prec, rec = getHist(densities_real, densities_fake)
    curve_pr = np.array([prec, rec]).T
    for scale in lambdas:
        density_real_scale = densities_real*scale
        density_fake_scale = densities_fake*(1/scale)
        diff_prec = scale - (np.sum(np.abs(density_real_scale - densities_fake)) / 2)
        diff_recall = (1/scale)  - (np.sum(np.abs(density_fake_scale - densities_real)) / 2)

        test = np.sum(np.minimum(density_real_scale, densities_fake))
        test2 = np.sum(np.minimum(densities_real, density_fake_scale))
        test3 = test / scale


        curve_pairs.append([np.clip(test, 0, 1), np.clip(test3,0,1)])

    curve_pairs = np.array(curve_pairs)
    diff = np.sum(prec - curve_pairs[:, 0])
    diff2 = np.sum(rec - curve_pairs[:, 1])
    print(diff, diff2)
    return curve_pr

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
        nearest_l1_distance = l1_distances_sorted[0]
        nearest_distance.append(nearest_l1_distance)
        row_check = (distances[:, 0] >= 0) & (distances[:, 1] >= 0)
        above_true = np.sum(row_check) > 0
        points_above.append(above_true)

    return np.array(points_above), np.array(nearest_distance)






