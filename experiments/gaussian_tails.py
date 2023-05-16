import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from experiments import experiment_calc as exp
from scipy.spatial import distance
import experiments.experiment_visualization as exp_vis

def resample(needed_samples, outlier_distance, dimension):
    new_samples = []
    while len(new_samples) < needed_samples:
        mean_vec = np.zeros(dimension)
        identity_cov = np.eye(dimension)
        real_features = np.random.multivariate_normal(mean_vec, identity_cov, 5000)
        for vector in real_features:
            mal_distance = distance.mahalanobis(vector, mean_vec, identity_cov)
            if mal_distance > outlier_distance:
                new_samples.append(vector)
    new_samples = np.array(new_samples)
    return new_samples[:needed_samples, :]

def doLinePlot(pr_pairs, dc_pairs, k_values,
               save_path="", save=False):
    plt.figure()
    plt.subplot(2,2,1)
    exp_vis.plotLine(k_values, pr_pairs[:, 0], "precision")
    exp_vis.plotLine(k_values, dc_pairs[:, 0], "density")
    plt.legend()
    plt.subplot(2, 2, 2)
    exp_vis.plotLine(k_values, pr_pairs[:, 1], "recall")
    exp_vis.plotLine(k_values, dc_pairs[:, 1], "coverage")
    plt.legend()
    if save:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def doBarPlots(pr_pairs, dc_pairs, k_values,
               save_path="", save=False):
    plt.figure()
    plt.subplot(2, 2, 1)
    exp_vis.plotBars(k_values, pr_pairs, "precision", "recall")
    plt.subplot(2, 2, 2)
    exp_vis.plotBars(k_values, dc_pairs, "density", "coverage")
    if save:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


def getRows(sample_size, dimension, k_vals, pr_pairs, dc_pais, reversed=False):
    rows = []
    for i in range(k_vals.shape[0]):
        precision = pr_pairs[i, 0]
        recall = pr_pairs[i, 1]
        density = dc_pais[i, 0]
        coverage = dc_pais[i, 1]
        k_val = k_vals[i]
        row = [sample_size, dimension, k_val, precision, recall,  density, coverage, reversed]
        rows.append(row)
    return rows

def plotExtra(k_vals, pr_pairs, dc_pairs, data, other_data, save_path):
    doBarPlots(pr_pairs, dc_pairs, k_vals, save_path="", save=False)
    plt.subplot(2, 1, 2)
    exp_vis.plotDistributions(data, other_data, "", save_path, True)

def getData():
    sample_sizes = [2000]
    dimensions = [2, 8, 16, 32, 64, 512, 1024]
    dimensions = [2,  64]

    for sample_size in sample_sizes:
        k_vals = np.array([1, 7, 99, 499, 999])
        for index, dimension in enumerate(dimensions):
            mean_vec = np.zeros(dimension)
            identity_cov = np.eye(dimension)
            real_features = np.random.multivariate_normal(mean_vec, identity_cov, sample_size)
            distances = []
            for vector in real_features:
                mal_distance = distance.mahalanobis(vector, mean_vec, identity_cov)
                distances.append(mal_distance)
            distances = np.array(distances)
            outlier_value = np.quantile(distances, 0.95)
            inlier_value = np.quantile(distances, 0.75)
            inliers = real_features[distances < inlier_value, :]
            outliers = resample(inliers.shape[0], outlier_value, dimension)
            pr_pairs, dc_pairs = exp.getKNN(inliers, outliers, k_vals)
            pr_pairs2, dc_pairs2 = exp.getKNN(outliers, inliers, k_vals)

            save_path = f"../gaussian_outlier/s{sample_size}_d{dimension}.png"
            save_path_reversed = f"../gaussian_outlier/reversed_s{sample_size}_d{dimension}.png"
            if dimension == 2:
                plotExtra(k_vals, pr_pairs, dc_pairs, inliers, outliers, save_path)
                plotExtra(k_vals, pr_pairs2, dc_pairs2, outliers, inliers, save_path_reversed)
            else:
                doBarPlots(pr_pairs, pr_pairs, k_vals, save_path=save_path, save=True)
                doBarPlots(pr_pairs2, dc_pairs2, k_vals, save_path=save_path_reversed, save=True)

getData()
plt.show()
