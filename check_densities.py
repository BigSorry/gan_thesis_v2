import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from experiments import experiment_calc as exp
import experiments.experiment_visualization as exp_vis
from sklearn import metrics
import helper_functions as util

# Assume real and fake prior is equal
def multiGaus(real_data, fake_data, dimension, scale_params):
    mixture_data = np.concatenate([real_data, fake_data])
    mean_vec = np.zeros(dimension)
    cov_real = np.eye(dimension) * scale_params[0]
    cov_fake = np.eye(dimension) * scale_params[1]
    densities_real = multivariate_normal.pdf(mixture_data, mean=mean_vec, cov=cov_real)
    densities_fake = multivariate_normal.pdf(mixture_data, mean=mean_vec, cov=cov_fake)

    return densities_real, densities_fake
    
def getGaussian(sample_size, dimension, lambda_factors):
    mean_vec = np.zeros(dimension)
    cov_ref = np.eye(dimension) * lambda_factors[0]
    cov_scaled = np.eye(dimension) * lambda_factors[1]
    reference_distribution = np.random.multivariate_normal(mean_vec, cov_ref, sample_size)
    scaled_distributions = np.random.multivariate_normal(mean_vec, cov_scaled, sample_size)

    return reference_distribution, scaled_distributions

def filterFactors(iters, k_vals, sample_size, dimension, factors, filter_std, real_scaling=False):
    factors_saved = []
    for i in range(iters):
        for scale in factors:
            lambda_factors = [factors[0], scale]
            reference_distribution, scaled_distribution = getGaussian(sample_size, dimension, lambda_factors)
            if real_scaling:
                distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = util.getDistanceMatrices(scaled_distribution, reference_distribution)
                pr_pairs, dc_pairs = exp.getKNN(distance_matrix_real, distance_matrix_fake, distance_matrix_pairs, k_vals)
            else:
                distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = util.getDistanceMatrices(reference_distribution, scaled_distribution)
                pr_pairs, dc_pairs = exp.getKNN(distance_matrix_real, distance_matrix_fake, distance_matrix_pairs, k_vals)
            # Clip density
            dc_pairs = np.clip(dc_pairs, 0, 1)
            pr_std = np.std(pr_pairs[:, 0]) + np.std(pr_pairs[:, 1])
            dc_std = np.std(dc_pairs[:, 0]) + np.std(dc_pairs[:, 1])
            if pr_std > filter_std or dc_std > filter_std:
                factors_saved.append(scale)
            # curve_classifier, curve_var_dist = exp.getGroundTruth("gaussian", reference_distribution, scaled_distribution, lambda_factors)
            # auc = metrics.auc(np.round(curve_var_dist[:, 1], 2), np.round(curve_var_dist[:, 0], 2))
            # if auc < 0.95 and auc > 0.05:
            #     factors_saved.append(scale)

    return factors_saved
def filterValues(values, atol_val):
    new_values = [values[0]]
    for value in values[1:]:
        diff = np.abs(np.array(new_values) - value)
        close_zero = np.isclose(diff, [0], atol=atol_val)
        if np.sum(close_zero) == 0:
            new_values.append(value)
    sorted = np.round(-np.sort(-np.array(new_values)), 4)

    return sorted
def saveRatios(iters, k_vals, sample_size, dimension, ratios, filter_std, real_scaling=False):
    base_value = 1
    factors = [base_value] + list(base_value * ratios)
    factors = np.round(factors, 4)
    filtered_scales = filterFactors(iters, k_vals, sample_size, dimension, factors, filter_std, real_scaling)
    print("filtered")
    print(len(filtered_scales))
    print(filtered_scales)
    saving = True
    if saving:
        util.savePickle("d64_factors.pkl", filtered_scales)

def getK(sample_size, low_boundary=10, step_low=2, step_high=50):
    low_k = [i for i in range(1, low_boundary, step_low)]
    high_k = [i for i in range(low_boundary, sample_size, step_high)]
    #high_k = [50, 100, 500, 750, 999]
    all_k = low_k + high_k

    return all_k

def getTable():
    return

def plotCurve(results_dict, dimension, map_path, real_scaling):
    for key, info_dict in results_dict.items():
        lambda_factors = list(key)
        scale_ratio = lambda_factors[0] if real_scaling else lambda_factors[1]
        k_vals = info_dict["k_vals"]
        pr_pairs = info_dict["pr_pairs"]
        dc_pairs = info_dict["dc_pairs"]
        curve_classifier = info_dict["curve_classifier"]
        reference_distribution = info_dict["reference_distribution"]
        scaled_distribution = info_dict["scaled_distribution"]
        plt.figure()
        save_path = f"{map_path}ratio{scale_ratio}.png"
        if dimension == 2:
            plt.subplot(1, 2, 1)
            plt.title(f"Scale ratio is {scale_ratio}")
            exp_vis.plotTheoreticalCurve(curve_classifier, curve_classifier, lambda_factors, save=False)
            exp_vis.plotKNNMetrics(pr_pairs, k_vals, "PR_KNN", "black", "", save=False)
            exp_vis.plotKNNMetrics(dc_pairs, k_vals, "DC_KNN", "yellow", save_path, save=False)
            plt.subplot(1, 2, 2)
            if real_scaling:
                exp_vis.plotDistributions(scaled_distribution, reference_distribution, 1, -1, "", save_path, save=True)
            else:
                exp_vis.plotDistributions(reference_distribution, scaled_distribution, -1, 1, "", save_path, save=True)
        else:
            plt.title(f"Scale ratio is  {scale_ratio}")
            exp_vis.plotTheoreticalCurve(curve_classifier, curve_classifier, lambda_factors, save=False)
            exp_vis.plotKNNMetrics(pr_pairs, k_vals, "PR_KNN", "black", "", save=False)
            exp_vis.plotKNNMetrics(dc_pairs, k_vals, "DC_KNN", "yellow", save_path, save=True)

def doCalcs(sample_size, dimensions, real_scaling=False):
    k_vals = getK(sample_size, low_boundary=100, step_low=5, step_high=50)
    base_value = 1
    ratios = util.readPickle("d64_factors.pkl")
    results = {}
    for index, ratio in enumerate(ratios[1:]):
        if index % 1 == 0:
            scale = base_value*ratio
            lambda_factors = [base_value, scale]
            reference_distribution, scaled_distribution = getGaussian(sample_size, dimensions, lambda_factors)
            if real_scaling:
                lambda_factors = [scale, base_value]
                distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = util.getDistanceMatrices(scaled_distribution, reference_distribution)
                curve_classifier = exp.getCurveClassifier("gaussian", scaled_distribution, reference_distribution, lambda_factors)
                #curve_var_dist = exp.getCurveVarDistance("gaussian", scaled_distribution, reference_distribution, lambda_factors)
            else:
                distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = util.getDistanceMatrices(reference_distribution, scaled_distribution)
                curve_classifier = exp.getCurveClassifier("gaussian", reference_distribution, scaled_distribution, lambda_factors)
                #curve_var_dist = exp.getCurveVarDistance("gaussian", reference_distribution, scaled_distribution, lambda_factors)

            pr_pairs, dc_pairs = exp.getKNN(distance_matrix_real, distance_matrix_fake, distance_matrix_pairs, k_vals)
            lambda_factors = (lambda_factors[0], lambda_factors[1])
            results[lambda_factors] = {}
            results[lambda_factors]["k_vals"] = k_vals
            results[lambda_factors]["pr_pairs"] = pr_pairs
            results[lambda_factors]["dc_pairs"] = dc_pairs
            results[lambda_factors]["curve_classifier"] = curve_classifier
            results[lambda_factors]["reference_distribution"] = reference_distribution
            results[lambda_factors]["scaled_distribution"] = scaled_distribution

    return results

def runExperiment(real_scaling):
    iters = 1
    sample_size = 1000
    k_vals = [i for i in range(1, sample_size, 10)]
    k_vals = [1, sample_size-1]
    dimension = 2
    ratios = 10
    try_ratios = np.round(np.linspace(0.01, .99, ratios), 4)
    filter_std = 0.2
    if real_scaling:
        map_path = f"./gaussian_dimension/paper_img/d{dimension}_real/"
    else:
        map_path = f"./gaussian_dimension/paper_img/d{dimension}_fake/"

    saveRatios(iters, k_vals, sample_size, dimension, try_ratios, filter_std, real_scaling=real_scaling)
    results_dict = doCalcs(sample_size, dimension, real_scaling=real_scaling)
    plotCurve(results_dict, dimension, map_path, real_scaling=real_scaling)

runExperiment(real_scaling=True)
runExperiment(real_scaling=False)
