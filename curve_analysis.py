import numpy as np
import helper_functions as util
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import glob
from experiments import experiment_calc as exp
from scipy import integrate
import auc_pre_filter as auc_filter

def assignAUCGroup(dataframe, auc_filter):
    dataframe["auc_group"] = -1
    for auc_index, (auc_begin, auc_end) in enumerate(auc_filter):
        bool_array = (dataframe["auc_score"] >= auc_begin) & (dataframe["auc_score"] < auc_end)
        dataframe.loc[bool_array, "auc_group"] = auc_index

def filterGroupedData(group, best_mode):
    if best_mode:
        top_distance = group.nsmallest(1, "distance").loc[:, "distance"].max()
        boolean_filter = (group["distance"] <= top_distance) | (
            np.isclose(group["distance"], [top_distance], atol=1e-2))
    else:
        top_distance = group.nlargest(1, "distance").loc[:, "distance"].max()
        boolean_filter = (group["distance"] >= top_distance) | (
            np.isclose(group["distance"], [top_distance], atol=1e-2))
    filter_data = group.loc[boolean_filter, :]

    return filter_data, top_distance

def plotPoint(curve, group_data, filter_data, title_text):
    plt.figure()
    plt.scatter(curve[:, 1], curve[:, 0])
    precision = group_data["first_score"]
    recall = group_data["second_score"]
    plt.title(title_text)
    plt.scatter(recall, precision, color="black")
    precision = filter_data["first_score"]
    recall = filter_data["second_score"]
    plt.scatter(recall, precision, color="green")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    metric_name = title_text[0]
    plt.savefig(f"./gaussian_dimension/points_analysis/{metric_name}/{title_text}.png")
    plt.close()

def getGaussianDimension(sample_size, dimension, transform_dimensions, lambda_factors):
    mean_vec = np.zeros(dimension)
    cov_ref = np.eye(dimension) * lambda_factors[0]
    cov_scaled = np.eye(dimension)
    index_transformed = np.random.choice(cov_scaled.shape[0], transform_dimensions, replace=False)
    for index in index_transformed:
        cov_scaled[index, index] = lambda_factors[1]
    reference_distribution = np.random.multivariate_normal(mean_vec, cov_ref, sample_size)
    scaled_distributions = np.random.multivariate_normal(mean_vec, cov_scaled, sample_size)

    return reference_distribution, scaled_distributions

def getDataframe(dataframe, real_scaled):
    column_names = ["metric_name", "scaling_mode", "iter", "dimension", "dimension_transformed",
                    "auc_score", "ratio", "k_val", "distance", "first_score", "second_score"]
    iters = 1
    sample_size = 1000
    dimensions = sorted(dataframe["dimension"].unique())
    #dimensions = [2, 512]
    dimension_pre_filter = dataframe["dimension"].unique()
    dimensions_transformed_all = dataframe["dimensions_transformed"].unique()
    dim_to_dim_transformed = {dimension_pre_filter[i]: dimensions_transformed_all[i] for i in range(dimension_pre_filter.shape[0])}
    k_vals = [i for i in range(1, sample_size, 1)]
    all_rows = []
    for index, dimension in enumerate(dimensions):
        sel_data = dataframe.loc[dataframe["dimension"] == dimension, :]
        ratios = sel_data["ratio"].unique()
        #ratios = all_ratios[np.random.choice(len(all_ratios), size=30, replace=False)]
        dimensions_transformed = dim_to_dim_transformed[dimension]
        row_results, theoretical_curves = getEvaluationPairs(iters, k_vals, sample_size, dimension, dimensions_transformed, ratios, real_scaled)
        all_rows.extend(row_results)
        all_data = pd.DataFrame(data=row_results, columns=column_names)
        grouped_data = all_data.groupby(["metric_name","dimension", "iter", "auc_score", "ratio"])
        for name, group in grouped_data:
            ratio = name[4]
            if ratio in theoretical_curves:
                curve = theoretical_curves[ratio]
                filter_data, top_distance = filterGroupedData(group, True)
                plotPoint(curve, group, filter_data, name)
            else:
                print(name)
    dataframe = pd.DataFrame(data=all_rows, columns=column_names)
    return dataframe

def getEvaluationPairs(iters, k_vals, sample_size, dimension, dimension_transformed, ratios, real_scaling=False):
    base_value = 1
    row_info = []
    scaling_mode = "real_scaled" if real_scaling else "fale_scaled"
    curves = {}
    for iter in range(iters):
        for index, ratio in enumerate(ratios):
            ratio = np.round(ratio, 4)
            scale = base_value*ratio
            lambda_factors = [base_value, scale]
            reference_distribution, scaled_distribution = getGaussianDimension(sample_size, dimension, dimension_transformed, lambda_factors)
            if real_scaling:
                lambda_factors = [scale, base_value]
                distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = util.getDistanceMatrices(scaled_distribution, reference_distribution)
                curve_classifier = exp.getCurveClassifier("gaussian", scaled_distribution, reference_distribution, lambda_factors)
            else:
                distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = util.getDistanceMatrices(reference_distribution, scaled_distribution)
                curve_classifier = exp.getCurveClassifier("gaussian", reference_distribution, scaled_distribution, lambda_factors)

            auc = integrate.trapz(np.round(curve_classifier[:, 1], 2), np.round(curve_classifier[:, 0], 2))
            pr_pairs, dc_pairs = exp.getKNN(distance_matrix_real, distance_matrix_fake, distance_matrix_pairs, k_vals)
            pr_aboves, pr_nearest_distances = exp.getStats(curve_classifier, pr_pairs)
            dc_aboves, dc_nearest_distances = exp.getStats(curve_classifier, dc_pairs)
            curves[ratio] = curve_classifier

            for index, k_val in enumerate(k_vals):
                pr_score = pr_pairs[index, :]
                dc_score = dc_pairs[index, :]
                pr_distance = pr_nearest_distances[index]
                dc_distance = dc_nearest_distances[index]
                pr_row = ["pr", scaling_mode, iter, dimension, dimension_transformed, auc, ratio, k_val, pr_distance, pr_score[0], pr_score[1]]
                dc_row = ["dc", scaling_mode, iter, dimension, dimension_transformed, auc, ratio, k_val, dc_distance, dc_score[0], dc_score[1]]
                row_info.append(pr_row)
                row_info.append(dc_row)

    return row_info, curves

df_path = "./dataframe_factors/dataframe_real.pkl"
factors_saved = pd.read_pickle(df_path)
dataframe_filtered = auc_filter.filterByAUC(factors_saved)
dataframe_factors_sel = dataframe_filtered.loc[dataframe_filtered["iter"] == 0, :]
getDataframe(dataframe_factors_sel, True)