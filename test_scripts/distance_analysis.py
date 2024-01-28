import datetime
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import integrate
from utility_scripts import helper_functions as util
from create_data_scripts import check_densities as ch_den
from experiments import experiment_calc as exp
from figure_scripts import create_boxplot as create_box

def filterByAUC(dataframe):
    dimensions = dataframe["dimension"].unique()
    all_ids = []
    for dim in dimensions:
        dimension_data = dataframe.loc[dataframe["dimension"] == dim, :]
        grouped = dimension_data.groupby(["dimensions_transformed"])["auc"].apply(
            lambda x: x.quantile(.75) - x.quantile(0.25))
        max_quantile_group = grouped.idxmax()
        filter_data = dimension_data.loc[dimension_data["dimensions_transformed"] == max_quantile_group, :]
        all_ids.extend(list(filter_data.index))

    filtered_dataframe = dataframe.loc[dataframe.index.isin(all_ids), :]
    filtered_dataframe.boxplot(column=['auc'], by=["dimension", "dimensions_transformed"])
    return filtered_dataframe

def plotDict(plot_dict, map_path, title_text):
    rows = len(plot_dict)
    print(rows)
    index = 1
    for k_value, info_dict in plot_dict.items():
        distances = info_dict["distance"]
        auc_scores = info_dict["auc"]
        plt.scatter(auc_scores, distances)
        index += 1

    save_path = f"{map_path}/{title_text}"
    plt.savefig(save_path)
    plt.close()

def scatterPlot(auc_scores, distances, map_path, title_text):
    plt.figure()
    plt.title(title_text)
    plt.scatter(auc_scores, distances)
    Path(map_path).mkdir(parents=True, exist_ok=True)
    save_path = f"{map_path}/{title_text}"
    plt.savefig(save_path)
    plt.close()


def plotDistances(sel_data, title_text, map_path):
    grouped_data = sel_data.groupby(["iter", "dimension", "auc_score"])
    auc_scores = []
    iqr_distances = []
    for group_name, group_data in grouped_data:
        distances = group_data['distance']
        auc_score = group_name[2]
        iqr_range = (distances.quantile(.25) - distances.quantile(0))
        iqr_distances.append(iqr_range)
        auc_scores.append(auc_score)

    scatterPlot(auc_scores, iqr_distances, map_path, title_text)

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


def getEvaluationPairs(iters, k_vals, sample_size, dimension, dimension_transformed, ratios, real_scaling=False):
    base_value = 1
    row_info = []
    scaling_mode = "real_scaled" if real_scaling else "fale_scaled"
    for iter in range(iters):
        for ratio_index, ratio in enumerate(ratios):
            scale = base_value*ratio
            lambda_factors = [base_value, scale]
            reference_distribution, scaled_distribution = ch_den.getGaussianDimension(sample_size, dimension, dimension_transformed, lambda_factors)
            if real_scaling:
                lambda_factors = [scale, base_value]
                distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = util.getDistanceMatrices(scaled_distribution, reference_distribution)
                curve_classifier = exp.getCurveClassifier("gaussian", scaled_distribution, reference_distribution, lambda_factors)
            else:
                distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = util.getDistanceMatrices(reference_distribution, scaled_distribution)
                curve_classifier = exp.getCurveClassifier("gaussian", reference_distribution, scaled_distribution, lambda_factors)

            auc = integrate.trapz(np.round(curve_classifier[:, 1], 2), np.round(curve_classifier[:, 0], 2))
            pr_pairs, dc_pairs = exp.getKNN(distance_matrix_real, distance_matrix_fake, distance_matrix_pairs, k_vals)
            pr_aboves, pr_nearest_distances = getStats(curve_classifier, pr_pairs)
            dc_aboves, dc_nearest_distances = getStats(curve_classifier, dc_pairs)

            for index, k_val in enumerate(k_vals):
                pr_score = pr_pairs[index, :]
                dc_score = dc_pairs[index, :]
                pr_distance = pr_nearest_distances[index]
                dc_distance = dc_nearest_distances[index]
                pr_above = pr_aboves[index]
                dc_above = dc_aboves[index]
                pr_row = [ratio_index, "pr", scaling_mode, iter, dimension, dimension_transformed, auc, k_val, pr_distance, pr_score[0], pr_score[1], pr_above]
                dc_row = [ratio_index, "dc", scaling_mode, iter, dimension, dimension_transformed, auc, k_val, dc_distance, dc_score[0], dc_score[1], dc_above]
                row_info.append(pr_row)
                row_info.append(dc_row)

    return row_info

def getDistances(dataframe_factors, dimensions, real_scaled, save_map):
    iters = 1
    sample_size = 1000
    column_names = ["ratio_index", "metric_name", "scaling_mode", "iter", "dimension", "dimension_transformed",
                    "auc_score", "k_val", "distance", "first_score", "second_score", "above_line"]
    dimension_pre_filter = dataframe_factors["dimension"].unique()
    dimensions_transformed_all = dataframe_factors["dimensions_transformed"].unique()
    dim_to_dim_transformed = {dimension_pre_filter[i]: dimensions_transformed_all[i] for i in
                              range(dimension_pre_filter.shape[0])}
    ratios = dataframe_factors["ratio"].unique()
    k_vals = [i for i in range(1, sample_size, 1)]
    all_rows = []
    for index, dimension in enumerate(dimensions):
        dimensions_transformed = dim_to_dim_transformed[dimension]
        dimension_rows = getEvaluationPairs(iters, k_vals, sample_size, dimension,
                                                   dimensions_transformed, ratios, real_scaled)
        all_rows.extend(dimension_rows)
        print(dimension)

    dataframe = pd.DataFrame(data=all_rows, columns=column_names)
    sub_map = "real_scaled/" if real_scaled else "fake_scaled/"
    time = datetime.datetime.today().strftime('%Y-%m-%d')
    dataframe.to_pickle(f"{save_map}{sub_map}dataframe_{dimensions}_{time}.pkl")

def experimentSeperation(dataframe):
    grouped_data = dataframe.groupby(["ratio_index", "iter", "dimension", "auc_score"])
    above_only_ids = []
    other_ids = []
    for group_key, group_data in grouped_data:
        above_percentage = np.mean(group_data["above_line"])
        dataframe_ids = [index for index in group_data.index]
        if above_percentage == 1:
            above_only_ids.extend(dataframe_ids)
        else:
            other_ids.extend(dataframe_ids)

    above_df = dataframe[dataframe.index.isin(above_only_ids)]
    other_df = dataframe[dataframe.index.isin(other_ids)]

    return above_df, other_df

def savePlots(metrics, scalings, save_path, best_mode=True):
    # save sub map if needed
    best_str = "best" if best_mode else "worst"
    map_path = f"{save_path}pick_{best_str}/"
    Path(map_path).mkdir(parents=True, exist_ok=True)
    k_val_bins = [1, 11, 101, 1000]
    auc_bins = [0, 0.3, 0.7, 0.99, 1.1]
    for scaling_mode in scalings:
        df_path_combined = f"{map_path}{scaling_mode}/current.pkl"
        all_dfs = util.readPickle(df_path_combined)
        for metric_name in metrics:
            metric_df = all_dfs.loc[all_dfs["metric_name"] == metric_name, :]
            sel_data = metric_df.loc[(metric_df["dimension"] > 1) & (metric_df["dimension"] <= 555), :]
            above_df, other_df = experimentSeperation(sel_data)
            print(above_df.shape[0] / sel_data.shape[0], other_df.shape[0] / sel_data.shape[0])
            dataframes_dict = {"above": above_df.copy(), "other": other_df.copy()}
            for key_name, df in dataframes_dict.items():
                df["k_groups"] = pd.cut(df["k_val"], k_val_bins, include_lowest=True,
                                                         right=False)
                df["auc_groups"] = pd.cut(df["auc_score"], auc_bins, include_lowest=True,
                                                           right=False)
                figure_map_path = f"./above_curve_analysis/{key_name}"
                create_box.getCorrelations(df, metric_name, scaling_mode, figure_map_path)
                create_box.distanceBoxplot(df, metric_name, scaling_mode, figure_map_path)
                create_box.distanceAUCBoxplot(df, metric_name, scaling_mode, figure_map_path)
                create_box.distanceDimensionBoxplot(df, metric_name, scaling_mode, figure_map_path)


def makeTestDF():
    dimensions = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    real_scaled = True
    save_map = "../dataframe_eval_test/"
    if real_scaled:
         path_factors = "../dataframe_factors/dataframe_real.pkl"
    else:
        path_factors = "../dataframe_factors/dataframe_fake.pkl"

    dataframe_factors = pd.read_pickle(path_factors)
    dataframe_factors_filtered = filterByAUC(dataframe_factors)
    getDistances(dataframe_factors_filtered, dimensions, real_scaled, save_map)
    real_scaled = False
    getDistances(dataframe_factors_filtered, dimensions, real_scaled, save_map)

def analyzeDF():
    metrics = ["pr", "dc"]
    scalings = ["real_scaled", "fake_scaled"]
    scalings = ["real_scaled"]
    save_map = "../dataframe_eval_test/"

    savePlots(metrics, scalings, save_map, best_mode=True)

makeTestDF()
#analyzeDF()