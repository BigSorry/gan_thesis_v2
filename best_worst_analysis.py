import numpy as np
import helper_functions as util
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.transforms
import pandas as pd
import seaborn as sns
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
    plt.savefig(save_path, bbox_inches="tight")
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

    return filter_data, top_distance

def plotAUCDistribution(dataframe, save_path):
    dataframe.boxplot(column=["auc_score"], by=["dimension"])
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def groupImage(dataframe, save_path):

    auc_sorted = sorted(dataframe["auc_score"].unique())
    dimensions_sorted = sorted(dataframe["dimension"].unique())
    # for auc in auc_sorted:
    #     auc_data = dataframe.loc[dataframe["auc_group"] == auc, :]
    k_vals_best = {}
    k_vals_worst = {}
    auc_scores_best = {}
    auc_scores_worst = {}
    for dimension in dimensions_sorted:
        if dimension not in k_vals_best:
            k_vals_best[dimension] = []
            k_vals_worst[dimension] = []
            auc_scores_best[dimension] = []
            auc_scores_worst[dimension] = []
        dimension_data = dataframe.loc[dataframe["dimension"] == dimension]
        grouped_data = dimension_data.groupby(["iter", "auc_score"])

        for key, group in grouped_data:
            filter_data, top_distance = filterGroupedData(group, True)
            filter_data_worst, top_distance_worst = filterGroupedData(group, False)
            best_picks = filter_data["k_val"].values
            worst_picks = filter_data_worst["k_val"].values
            median_value = np.median(best_picks)
            auc_score = key[1]
            if median_value <= 10:
                k_vals_best[dimension].extend(best_picks)
                auc_scores_best[dimension].append(auc_score)
            else:
                k_vals_worst[dimension].extend(worst_picks)
                auc_scores_worst[dimension].append(auc_score)

    Path(save_path).mkdir(parents=True, exist_ok=True)
    plotBoxplot(k_vals_best, [1, 11], "K-value", save_path+"best.png")
    plotBoxplot(k_vals_worst, [1, 999], "K-value", save_path+"worst.png")
    plotBoxplot(auc_scores_best, [0, 1.1], "Auc score", save_path+"auc_best.png")
    plotBoxplot(auc_scores_worst, [0, 1.1], "Auc score", save_path+"auc_worst.png")




def createPlots(metrics, scalings, auc_filter, save_map):
    read_all = False
    for scaling_mode in scalings:
        sub_map = f"{save_map}{scaling_mode}/"
        if read_all:
            df_path = f"./dataframe_evaluation/{scaling_mode}/*.pkl"
            df_list = []
            sum_rows = 0
            for file_name in glob.glob(df_path):
                df = pd.read_pickle(file_name)
                print(file_name, df.shape[0])
                sum_rows += df.shape[0]
                df_list.append(df)
            all_dfs = pd.concat(df_list, axis=0, ignore_index=True)
        else:
            df_path_combined = f"./dataframe_evaluation/{scaling_mode}/combined/dataframe_all.pkl"
            all_dfs = util.readPickle(df_path_combined)

        assignAUCGroup(all_dfs, auc_filter)
        for metric_name in metrics:
            metric_df = all_dfs.loc[all_dfs["metric_name"] == metric_name, :]
            plotAUCDistribution(metric_df, sub_map+"auc_all.png")
            groupImage(metric_df, sub_map)

metrics = ["pr"]
scalings = ["real_scaled", "fake_scaled"]
auc_filter = [(0, 0.3), (0.3, 0.7), (0.7, 1.1)]
auc_filter = [(0, 1.1)]
save_map = "./gaussian_dimension/best_worst_case/"
createPlots(metrics, scalings, auc_filter, save_map)
plt.show()