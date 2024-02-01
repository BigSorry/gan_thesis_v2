import numpy as np
from utility_scripts import helper_functions as util
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from utility_scripts import winner_pick_utility as win_util

def createLines(distance_median, minimum_distance, maximum_distance):
    x_vals = np.arange(distance_median.shape[0]) + 1
    plt.plot(x_vals, maximum_distance, label="75th quantile  distance", c="red")
    plt.plot(x_vals, distance_median, label="50th quantile distance", c="orange")
    plt.plot(x_vals, minimum_distance, label="25th quantile distance", c="green")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fontsize="small", ncol=3)

def createBoxplot(title_text, boxes, max_value=1):
    plt.title(title_text)
    start_index = 12
    high_indices = np.logspace(np.log10(start_index), np.log10(len(boxes) - 1), 20, dtype=int)
    # Make sure last box is the highest index
    high_indices[-1] = len(boxes) - 1
    #print(high_indices)
    all_indices = list(np.arange(10)) + list(high_indices)
    selected_boxes = boxes.iloc[all_indices]
    positions = np.array(all_indices) + 1

    w = 0.05
    width = lambda p, w: 10 ** (np.log10(p) + w / 2.) - 10 ** (np.log10(p) - w / 2.)
    plt.boxplot(selected_boxes, positions=positions, widths=width(positions, w))
    plt.xticks(positions, positions)
    plt.ylabel("distance")
    y_upper_limit = max(1, max_value) + 0.2
    plt.ylim([0, y_upper_limit])
    plt.xlabel("k-value")
    plt.xscale("log")

def saveBoxplots(grouped_data, parent_map_path, sub_map):
    Path(parent_map_path+sub_map).mkdir(parents=True, exist_ok=True)
    for group_name, group_df in grouped_data:
        print(group_name, group_df.shape[0])
        maximum_distance = group_df.groupby("k_val")["distance"].quantile(q=0.75)
        minimum_distance = group_df.groupby("k_val")["distance"].quantile(q=0.25)
        distance_median = group_df.groupby("k_val")["distance"].quantile(q=0.5)

        title_text = ""
        file_name = ""
        if "auc" in sub_map and "dimension" in sub_map:
            auc_interval = group_name[0]
            dimension = group_name[1]
            title_text = f"Dimension {dimension} and AUC group {auc_interval}"
            file_name = f"auc_{auc_interval.left}-{auc_interval.right}_dim_{dimension}"
        elif "auc" in sub_map:
            title_text = f"AUC group {group_name}"
            file_name = f"auc_{group_name}"
        elif "dimension" in sub_map:
            title_text = f"Dimension {group_name}"
            file_name = f"dimension_{group_name}"


        boxes = group_df.groupby("k_val").apply(lambda x: [x["distance"]])
        plt.figure(figsize=(6, 5))
        createBoxplot(title_text, boxes)
        createLines(distance_median, minimum_distance, maximum_distance)


        save__path = f"{parent_map_path+sub_map}{file_name}.png"
        plt.savefig(save__path, bbox_inches="tight", dpi=300)
        plt.close()

def noGrouping(dataframe, metric_name, scaling_mode, map_path):
    maximum_distance = dataframe.groupby("k_val")["distance"].quantile(q=0.75)
    minimum_distance = dataframe.groupby("k_val")["distance"].quantile(q=0.25)
    distance_median = dataframe.groupby("k_val")["distance"].quantile(q=0.5)

    scenario_str = f"{metric_name}_{scaling_mode}"
    title_text = scenario_str
    file_name = f"{scenario_str}.png"
    boxes = dataframe.groupby("k_val").apply(lambda x: [x["distance"]])
    createBoxplot(title_text, boxes)
    createLines(distance_median, minimum_distance, maximum_distance)

    save__path = f"{map_path}{file_name}"
    plt.savefig(save__path, dpi=300)
    plt.close()

def makeGroups(sel_df, map_path, metric_name, scaling_mode, one_var):
    if one_var == "auc_dimension":
        auc_dimension_grouped = sel_df.groupby(["auc_groups", "dimension"])
        saveBoxplots(auc_dimension_grouped, map_path, "/auc_dimension/")
    else:
        auc_grouped = sel_df.groupby(["auc_groups"])
        dimension_grouped = sel_df.groupby(["dimension"])
        saveBoxplots(auc_grouped, map_path, "/auc/")
        saveBoxplots(dimension_grouped, map_path, "/dimension/")
        noGrouping(sel_df, metric_name, scaling_mode, map_path)

def getFilterMask(dataframe, max_winner=1):
    grouped_data = dataframe.groupby(["iter", "dimension", "auc_score"])
    excluded_keys = []
    for (iter, dimension, auc_score), experiment_data in grouped_data:
        if experiment_data.shape[0] > 999:
            experiment_data = experiment_data.iloc[:999, :]
        filter_data, top_distance = win_util.filterGroupedData(experiment_data, True)
        winner_rows = filter_data.shape[0]
        if winner_rows > max_winner:
            excluded_keys.append([iter, dimension, auc_score])

    filter_mask = np.ones(dataframe.shape[0]).astype(bool)
    for (iter, dimension, auc_score) in excluded_keys:
        excluded = (dataframe["iter"] == iter) & (dataframe["dimension"] == dimension) & (dataframe["auc_score"] == auc_score)
        filter_mask[excluded] = False

    return filter_mask

def createPlots(metrics, scalings, max_winner, map_path, one_var="all"):
    # save sub map if needed
    auc_bins = [0, 0.3, 0.7, 0.9, 1.1]
    auc_bins = [0, .3, .7, 1.1]
    for scaling_mode in scalings:
        df_path_combined = f"../dataframe_evaluation/{scaling_mode}/combined/dataframe_all.pkl"
        all_dfs = util.readPickle(df_path_combined)
        for metric_name in metrics:
            sub_map = f"{map_path}{metric_name}_{scaling_mode}/"
            metric_df = all_dfs.loc[all_dfs["metric_name"] == metric_name, :]
            sel_data = metric_df.loc[(metric_df["dimension"] > 1) & (metric_df["dimension"] <= 555), :]
            filter_mask = getFilterMask(sel_data, max_winner)
            filtered_data = sel_data.loc[filter_mask, :].copy()
            filtered_data.loc[:, "auc_groups"] = pd.cut(filtered_data["auc_score"], auc_bins, include_lowest=True, right=False)

            print(sel_data.shape[0], filtered_data.shape[0])
            makeGroups(filtered_data, sub_map, metric_name, scaling_mode, one_var)

def allExperiments():
    metrics = ["pr", "dc"]
    scalings = ["real_scaled", "fake_scaled"]
    # Max 1 winner
    max_winners = [1]
    for max_winner in max_winners:
        overview_map_boxplots = f"./boxplot_all_k/max_winner{max_winner}/"
        createPlots(metrics, scalings, max_winner, overview_map_boxplots, one_var="all")

def smallExperiment():
    metrics = ["pr"]
    scalings = ["real_scaled"]
    max_winner = 1
    overview_map_boxplots = f"./boxplot_all_k/max_winner{max_winner}/"
    one_var = "auc_dimension"
    createPlots(metrics, scalings, max_winner, overview_map_boxplots, one_var)

allExperiments()
