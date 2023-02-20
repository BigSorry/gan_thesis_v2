import os
import numpy as np
import pandas as pd
import experiments_v2.helper_functions as util
import matplotlib.pyplot as plt
import metrics.likelihood_classifier as llc
from sklearn.metrics import pairwise_distances
import experiments_v2.curve.likelihood_estimations as ll_est
import visualize as plotting
import experiments_v2.helper_functions as util
import metrics.not_used.metrics as mtr
from sklearn.neighbors import KNeighborsClassifier

def getDistributions(sample_size, dimension, scale_param):
    distributions = []
    mean_vec = np.zeros(dimension)
    for scale in scale_param:
        cov_mat = np.eye(dimension)*scale
        samples = np.random.multivariate_normal(mean_vec, cov_mat, sample_size)
        distributions.append(samples)
    return distributions

def doKNNS(mixture_dict, test_data, k_vals, label_value):
    test_labels = np.array([label_value for i in range(test_data.shape[0])])
    error_rates = {}
    for ber_p, data_dict in mixture_dict.items():
        mixture_data = data_dict["train_data"]
        mixture_label = data_dict["train_label"]
        distance_matrix = util.getDistanceMatrix(mixture_data, mixture_data)
        distance_pairs = util.getDistanceMatrix(test_data, mixture_data)
        error_rates[ber_p] = {}
        for k_val in k_vals:
            boundaries_union = distance_matrix[:, k_val]
            row_form = np.expand_dims(boundaries_union, axis=0)
            truth_table = (distance_pairs < row_form)
            predictions = truth_table.any(axis=1).astype(int)
            fpr, fnr = llc.getScores(test_labels, predictions)
            error_rates[ber_p][k_val] = [fpr, fnr]

    return error_rates

def takeSamples(data, indices, p):
    sample_size = data.shape[0]
    taken_indices = int(sample_size * p)
    row_indices = np.random.choice(indices, taken_indices, replace=False)
    return data[row_indices, :]

def doMixture(real_data, fake_data):
    bernoulli_p = [0, 0.25, 0.5, 0.75, 1]
    indices = np.arange(real_data.shape[0])
    mixture_dict = {}
    for p in bernoulli_p:
        real_taken = takeSamples(real_data, indices, p)
        fake_taken = takeSamples(fake_data, indices, 1-p)
        mixture_data = np.concatenate([real_taken, fake_taken])
        mixture_labels = np.concatenate([np.ones(real_taken.shape[0]), np.zeros(fake_taken.shape[0])])
        mixture_dict[p] = {"train_data":mixture_data, "train_label":mixture_labels}

    return mixture_dict

def doPlotting(error_dict):
    fig, (ax1,ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    plt.title("Mixture Z= p.R + (1-p).F")
    ber_ps = list(error_dict.keys())
    plt.ylim([0, 1.1])
    x_axes = (np.arange(len(ber_ps))+1)*2
    i=2
    for ber_p, ber_dict in error_dict.items():
        scores = np.array(list(ber_dict.values()))
        fpr = scores[:, 0]
        fnr = scores[:, 1]
        ax1.boxplot(fpr, positions=[i])
        ax2.boxplot(fnr, positions=[i])
        i+=2


    ax1.set_ylabel("FPR")
    string_labels = [f"p={p}" for p in ber_ps]
    plt.xticks(x_axes, string_labels)
    ax1.set_xticks(x_axes)
    ax1.set_xticklabels(string_labels)
    ax1.set_xlim(0, np.max(x_axes)+2)


    ax2.set_ylabel("FNR")

def plotCurve(curve_classifier, curve_var_dist):
    plotting.plotCurve(curve_classifier, label_text="Likelihood ratio test")
    plotting.plotCurve(curve_var_dist, label_text="Variational distance")
    plt.legend()

def getGroundTruth(real_data, fake_data, scale_factors):
    lambdas = llc.getPRLambdas(angle_count=1000)
    densities_real, densities_fake = ll_est.getDensities(real_data, fake_data, scale_factors, method_name="multi_gaus")
    densities_real_norm = densities_real / np.sum(densities_real)
    densities_fake_norm = densities_fake / np.sum(densities_fake)

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

def showKNN(real_data, fake_data, k_vals):
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
def getCurves(real_data, fake_data):
    params = {"k_cluster": 20, "angles": 1001, "kmeans_runs": 2}
    pr_score, curve, cluster_labels = mtr.getHistoPR(real_data, fake_data, params)
    classifier = KNeighborsClassifier(n_neighbors=1)
    params = {"threshold_count": 100, "angles": 1001, "classifier": classifier}
    pr_score, curve2, prob_labels = mtr.getClassifierPR(real_data, fake_data, params)

    return [curve, curve2]

def plotTheoreticalCurve(curve_classifier, curve_var_dist, scale_factors, save=True):
    plt.title(f"Lambda scaling real cov {scale_factors[0]} and lambda scaling fake cov {scale_factors[1]}")
    plotCurve(curve_classifier, curve_var_dist)
    if save:
        path = f"C:/Users/lexme/Documents/gan_thesis_v2/present/1-02-23/ground-truths/scale_{scale_factors}.png"
        plt.savefig(path)

# Plotting is reversed to get recall on x axis
def plotKNNMetrics(pr_pairs, dc_pairs, pr_above, dc_above, k_values, save_path, save=True):
    annotate_text = [f"k={k}" for k in k_values]
    plt.scatter(pr_pairs[~pr_above, 1], pr_pairs[~pr_above, 0], c="red", label=f"Under_Precision_Recall_KNN")
    plt.scatter(pr_pairs[pr_above, 1], pr_pairs[pr_above, 0], c="green", label=f"Upper_Precision_Recall_KNN")
    plt.scatter(dc_pairs[~dc_above, 1], dc_pairs[~dc_above, 0], c="yellow", label=f"Under_Density_Coverage_KNN")
    plt.scatter(dc_pairs[dc_above, 1], dc_pairs[dc_above, 0], c="black", label=f"Upper_Density_Coverage_KNN")
    for index, text in enumerate(annotate_text):
        pr_coords = (pr_pairs[index, 1], pr_pairs[index, 0])
        dc_coords = (dc_pairs[index, 1], dc_pairs[index, 0])
        plotting.specialAnnotate(text, pr_coords)
        plotting.specialAnnotate(text, dc_coords)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=2)
    if save:
        plt.savefig(save_path)
        plt.close()
def plotCurveMetrics(histo_method, classifier_method, scale_factors, save=True):
    plt.scatter(histo_method[:, 1], histo_method[:, 0], c="green", label=f"Precision_Recall_Histo_Curve")
    plt.scatter(classifier_method[:, 1], classifier_method[:, 0], c="black", label=f"Precision_Recall_Class_Curve")
    plt.legend()
    if save:
        path = f"C:/Users/lexme/Documents/gan_thesis_v2/plot_paper/gaussian/scale_{scale_factors}.png"
        plt.savefig(path)
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

def plotStats(pr_above, dc_above, save_path, save=False):
    pr_mean = pr_above.mean()
    dc_mean = dc_above.mean()
    ax = ["pr", "dc"]
    plt.bar(ax, [pr_mean, dc_mean], color='black', width=0.25)
    plt.ylim([0, 1.1])
    plt.xlabel("Percentage points above the curve")
    if save:
        plt.savefig(save_path)
        plt.close()

def makeMap(sample_size, dimension, percentage, other_percentage):
    split_val = [0.25, 0.5, 0.75, 1]
    max_percentage = max(percentage, other_percentage)
    begin = 0
    end=0
    for val in split_val:
        if max_percentage <= val:
            begin = val - 0.25
            end = val
            break
    pc_save_map = f"C:/Users/Lex/Documents/gan_thesis_v2/plot_paper/gaussian/" \
                  f"{begin}-{end}/" \
                  f"s{sample_size}_d{dimension}/"
    pc_save_map = f"C:/Users/lexme/Documents/gan_thesis_v2/plot_paper/gaussian/" \
                  f"{begin}-{end}/" \
                  f"s{sample_size}_d{dimension}/"
    if not os.path.isdir(pc_save_map):
        os.makedirs(pc_save_map)

    return pc_save_map


def doExpiriment(sample_size, dimension):
    var_factors = [0.01, 0.1, 0.2,  0.25, 0.5, 1]
    var_factors = [0.01, 0.1, 0.2, 0.25, 0.5, 0.75, 1]
    curve_methods = False
    knn_methods = True
    row_values = []
    show_plots=False
    save_plots=False
    for factor in var_factors:
        for other_factor in var_factors:
            scale_factors = [factor, other_factor]
            distributions = getDistributions(sample_size, dimension, scale_factors)
            real_data = distributions[0]
            fake_data = distributions[1]
            curve_classifier, curve_var_dist = getGroundTruth(real_data, fake_data, scale_factors)

            k_vals = np.array([1, 2, 3, 4, 8, 9, 16, 32, 64, sample_size - 1])
            #k_vals = np.arange(1, sample_size, 5)

            if knn_methods:
                pr_pairs, dc_pairs = showKNN(real_data, fake_data, k_vals)
                pr_above, pr_nearest_distances = getStats(curve_var_dist, pr_pairs)
                dc_above, dc_nearest_distances = getStats(curve_var_dist, dc_pairs)

                pc_save_map = makeMap(sample_size, dimension, pr_above.mean(), dc_above.mean())
                values = [factor, other_factor, pr_above.mean(), dc_above.mean(), pr_nearest_distances.mean(), dc_nearest_distances.mean()]
                row_values.append(values)

                if show_plots:
                    plt.figure(figsize=(12, 10))
                    save_path = f"{pc_save_map}params_r{scale_factors[0]}_f{scale_factors[1]}.png"
                    plt.subplot(1, 3, 1)
                    plotTheoreticalCurve(curve_classifier, curve_var_dist, scale_factors, save=False)
                    plotKNNMetrics(pr_pairs, dc_pairs, pr_above, dc_above, k_vals, save_path, save=False)
                    plt.subplot(1, 3, 2)
                    plotting.plotDistributions(real_data, fake_data, "", save_path, save=False)
                    plt.subplot(1, 3, 3)
                    plotStats(pr_above, dc_above, save_path, save=save_plots)
                    plt.show()

            if curve_methods:
                pr_curve_histo, pr_curve_class = getCurves(real_data, fake_data)
                plotCurveMetrics(pr_curve_histo, pr_curve_class, scale_factors, save=False)

    columns= ["real_lambda_factor", "fake_lambda_factor", "pr_under_mean", "dc_under_mean",
              "pr_nearest_distances", "dc_nearest_distances"]
    result_dataframe = pd.DataFrame(data=row_values, columns=columns)
    pr_pivot = result_dataframe.pivot(index="real_lambda_factor", columns="fake_lambda_factor", values="pr_nearest_distances")
    dc_pivot = result_dataframe.pivot(index="real_lambda_factor", columns="fake_lambda_factor", values="dc_nearest_distances")
    plotting.HeatMapPivot(pr_pivot, title_text=f"Precision and Recall, samples{sample_size}_dimension{dimension}",
                          save=True, save_path=f"./images/pr_s{sample_size}_d{dimension}.png")
    plotting.HeatMapPivot(dc_pivot, title_text=f"Density and Coverage, samples{sample_size}_dimension{dimension}",
                          save=True, save_path=f"./images/dc_s{sample_size}_d{dimension}.png")

samples=1000
doExpiriment(sample_size=samples, dimension=2)
doExpiriment(sample_size=samples, dimension=16)
doExpiriment(sample_size=samples, dimension=64)

samples=3000
doExpiriment(sample_size=samples, dimension=2)
doExpiriment(sample_size=samples, dimension=16)
doExpiriment(sample_size=samples, dimension=64)

samples=5000
doExpiriment(sample_size=samples, dimension=2)
doExpiriment(sample_size=samples, dimension=16)
doExpiriment(sample_size=samples, dimension=64)
plt.show()
