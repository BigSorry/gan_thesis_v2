import numpy as np
from utility_scripts import helper_functions as util
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import glob

def plotBoxplot(k_vals_best, ylim_arr, ylabel_text, save_path):
    plt.figure()
    dimensions = list(k_vals_best.keys())
    old_x = np.arange(len(dimensions)) + 1
    values = list(k_vals_best.values())

    plt.boxplot(values)
    plt.xlabel("Dimensions")
    plt.xticks(old_x, dimensions)
    plt.ylabel(ylabel_text)
    plt.ylim(ylim_arr)
    plt.yscale("symlog")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plotDistances(distance_dict, distance_dict_worst, auc_dict, save_path):
    for dimension, distance in distance_dict.items():
        plt.figure()
        x_values = auc_dict[dimension]
        y_values = distance
        plt.scatter(x_values, y_values, label="Best distances")
        y_values = distance_dict_worst[dimension]
        plt.scatter(x_values, y_values, label="Worst distances")

        plt.legend()
        plt.xlabel("Auc score")
        plt.ylabel("L1 distance")
        plt.savefig(save_path+f"_dim_{dimension}.png", bbox_inches="tight")
        plt.close()

def assignAUCGroup(dataframe, auc_filter):
    dataframe["auc_group"] = -1
    for auc_index, (auc_begin, auc_end) in enumerate(auc_filter):
        bool_array = (dataframe["auc_score"] >= auc_begin) & (dataframe["auc_score"] <= auc_end)
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

    return filter_data

def plotAUCDistribution(dataframe, save_path):
    dataframe.boxplot(column=["auc_score"], by=["dimension"])
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def groupImage(dataframe, save_path):
    grouped_data = dataframe.groupby(["iter", "auc_score", "dimension"])
    best_distances = {}
    worst_distances = {}
    auc_scores = {}
    for experiment_ids, experiment_data in grouped_data:
        distances = experiment_data["distance"]
        worst_distance = distances.max()
        best_distance = distances.min()

        auc_score = experiment_ids[1]
        dimension = experiment_ids[2]

        if dimension not in best_distances:
            best_distances[dimension] = [best_distance]
            worst_distances[dimension] = [worst_distance]
            auc_scores[dimension] = [auc_score]
        else:
            best_distances[dimension].append(best_distance)
            worst_distances[dimension].append(worst_distance)
            auc_scores[dimension].append(auc_score)

    Path(save_path).mkdir(parents=True, exist_ok=True)
    plotDistances(best_distances, worst_distances, auc_scores, save_path)
    #plotBoxplot(best_distances, [0, 999], "K-value", save_path+"best.png")
    #plotBoxplot(worst_distances, [0, 999], "K-value", save_path+"worst.png")
    #plotBoxplot(auc_scores_best, [0, 1.1], "Auc score", save_path+"auc_best.png")
    #plotBoxplot(auc_scores_worst, [0, 1.1], "Auc score", save_path+"auc_worst.png")

def createPlots(metrics, scalings, save_map):
    auc_bins = [0, 0.3, 0.7, 0.9, 1.1]
    for scaling_mode in scalings:
        df_path_combined = f"../dataframe_evaluation/{scaling_mode}/combined/dataframe_all.pkl"
        all_dfs = util.readPickle(df_path_combined)
        for metric_name in metrics:
            sub_map = f"{save_map}{metric_name}_{scaling_mode}/"
            metric_df = all_dfs.loc[all_dfs["metric_name"] == metric_name, :]
            sel_data = metric_df.loc[(metric_df["dimension"] > 1) & (metric_df["dimension"] <= 555), :]
            sel_data.loc[:, "auc_groups"] = pd.cut(sel_data["auc_score"], auc_bins, include_lowest=True, right=False)

            groupImage(sel_data, sub_map)
            plotAUCDistribution(sel_data, sub_map+"auc_all.png")

metrics = ["pr", "dc"]
scalings = ["real_scaled", "fake_scaled"]
save_map = "./best_worst_case/"
createPlots(metrics, scalings, save_map)
