import numpy as np
from utility_scripts import helper_functions as util
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


def saveBoxplot(boxes, k_groups, text_title, sub_map, file_name):
    plt.title(text_title)
    plt.boxplot(boxes)
    plt.ylim([0, 1.1])
    samples_boxes = [len(box) for box in boxes]
    new_xticks = [f"k_vals={k_group.left}-{k_group.right}\n samples={samples_boxes[i]}" for i, k_group in enumerate(k_groups)]
    plt.xticks(np.arange(len(boxes)) + 1, new_xticks)

    Path(sub_map).mkdir(parents=True, exist_ok=True)
    save_path = f"{sub_map}/{file_name}"
    plt.savefig(save_path)
    plt.close()

def distanceBoxplot(dataframe, metric_name, scaling_mode, map_save_path):
    # No auc grouping
    k_groups = dataframe["k_groups"].unique()
    boxes = []
    total_samples = 0
    for k_group in k_groups:
        sel_data = dataframe.loc[dataframe["k_groups"] == k_group, :]
        total_samples += sel_data.shape[0]
        boxes.append(sel_data["distance"])

    sub_map = f"{map_save_path}/{metric_name}/{scaling_mode}/"
    file_name = f"k_val grouping only.png"
    title_text = f"All auc, samples {total_samples}"
    saveBoxplot(boxes, k_groups, title_text, sub_map, file_name)

def distanceAUCBoxplot(dataframe, metric_name, scaling_mode, map_save_path):
    k_groups = dataframe["k_groups"].unique()
    grouped_auc = dataframe.groupby("auc_groups")
    for group_name, group_data in grouped_auc:
        interval_auc = group_name
        boxes = []
        for k_group in k_groups:
            sel_data = group_data.loc[dataframe["k_groups"] == k_group, :]
            boxes.append(sel_data["distance"])\

        title_text = f"Auc{interval_auc.left}-{interval_auc.right}"
        sub_map = f"{map_save_path}/{metric_name}/{scaling_mode}/auc/"
        file_name = f"auc{interval_auc.left}-{interval_auc.right}.png"
        saveBoxplot(boxes, k_groups, title_text, sub_map, file_name)

def distanceDimensionAUCBoxplot(dataframe, metric_name, scaling_mode, map_save_path):
    k_groups = dataframe["k_groups"].unique()
    grouped_auc = dataframe.groupby(["auc_groups", "dimension"])
    for group_name, group_data in grouped_auc:
        interval_auc = group_name[0]
        dimension = group_name[1]
        boxes = []
        for k_group in k_groups:
            sel_data = group_data.loc[dataframe["k_groups"] == k_group, :]
            boxes.append(sel_data["distance"])\

        title_text = f"Dim{dimension} and auc{interval_auc.left}-{interval_auc.right}"
        sub_map = f"{map_save_path}/{metric_name}/{scaling_mode}/auc_dimension/"
        file_name = f"auc{interval_auc.left}-{interval_auc.right}_dim{dimension}.png"
        saveBoxplot(boxes, k_groups, title_text, sub_map, file_name)

def distanceDimensionBoxplot(dataframe, metric_name, scaling_mode, map_save_path):
    k_groups = dataframe["k_groups"].unique()
    grouped_dim = dataframe.groupby(["dimension"])
    for dimension, group_data in grouped_dim:
        boxes = []
        for k_group in k_groups:
            sel_data = group_data.loc[dataframe["k_groups"] == k_group, :]
            boxes.append(sel_data["distance"])\

        title_text = f"Dim{dimension}"
        sub_map = f"{map_save_path}/{metric_name}/{scaling_mode}/dimension/"
        file_name = f"dim{dimension}.png"
        saveBoxplot(boxes, k_groups, title_text, sub_map, file_name)

def getPercentageAUCOnly(dataframe_top_picks):
    # Full count without taking into percentages into consideration
    # Also add union count for all auc groups as last row
    remaining_results = dataframe_top_picks.loc[dataframe_top_picks["k_value"] >= 1, :]
    percentage_count = remaining_results.groupby(["auc_groups", "k_groups"])["count_value"].sum().unstack()
    percentage_count.loc[pd.Interval(left=0, right=1.1)] = percentage_count.sum(axis=0)
    total_dim_count = percentage_count.sum(axis=1)
    percentage_count_div = (percentage_count.div(total_dim_count.values, axis=0) * 100).round(decimals=2)

    return percentage_count_div

def getPercentageCount(dataframe_top_counts, auc_begin, auc_end):
    dataframe_top_counts_sel = dataframe_top_counts.loc[(dataframe_top_counts["auc_score"] > auc_begin) & (dataframe_top_counts["auc_score"] <= auc_end), :]
    remaining_results = dataframe_top_counts_sel.loc[dataframe_top_counts_sel["k_value"] >= 1, :]
    # Get amount of experiments in a certain auc range
    percentage_count = remaining_results.groupby(["dimension", "k_groups"])["count_value"].sum().unstack()
    total_dim_count = percentage_count.sum(axis=1)
    percentage_count_div = (percentage_count.div(total_dim_count.values, axis=0) * 100).round(decimals=2)

    return percentage_count_div

def getCorrelations(dataframe, metric_name, scaling_mode, map_save_path):
    dimensions = dataframe["dimension"].unique()
    k_groups = dataframe["k_groups"].unique()
    auc_groups = sorted(dataframe["auc_groups"].unique(), key=lambda x : x.left)
    for dimension in dimensions:
        plt.figure()
        fig, axs = plt.subplots(k_groups.shape[0], 1, sharex=True, sharey=True)
        fig.suptitle(f"Dimension {dimension}")
        for plot_index, k_group in enumerate(k_groups):
            sel_data = dataframe.loc[(dataframe["dimension"] == dimension) & (dataframe["k_groups"] == k_group), :]
            boxes = []
            for auc_group in auc_groups:
                data = sel_data.loc[(dataframe["auc_groups"] == auc_group), :]
                if data.shape[0] == 0:
                    boxes.append([])
                    #print("empty", dimension, k_group, auc_group)
                else:
                    boxes.append(data["distance"])

            # TODO Refactoring
            axs[plot_index].boxplot(boxes)
            axs[plot_index].set_title(f"K-value group [{k_group.left}-{k_group.right - 1}]")

        # Create figure
        new_xticks = [f"AUC={auc_interval.left}-{auc_interval.right}\n" for i, auc_interval in
                      enumerate(auc_groups)]
        plt.xticks(np.arange(len(auc_groups)) + 1, new_xticks)
        sub_map = f"{map_save_path}/{metric_name}/{scaling_mode}/auc_dimension/"
        file_name = f"dim{dimension}.png"
        Path(sub_map).mkdir(parents=True, exist_ok=True)
        save_path = f"{sub_map}/{file_name}"
        fig.tight_layout()
        plt.savefig(save_path)
        plt.close()

def createPlots(metrics, scalings, save_path):
    # save sub map if needed
    Path(save_path).mkdir(parents=True, exist_ok=True)
    k_val_bins = [1, 11, 101, 1000]
    auc_bins = [0, 0.3, 0.7, 0.9, 1.1]
    for scaling_mode in scalings:
        df_path_combined = f"./dataframe_evaluation/{scaling_mode}/combined/dataframe_all.pkl"
        all_dfs = util.readPickle(df_path_combined)
        for metric_name in metrics:
            metric_df = all_dfs.loc[all_dfs["metric_name"] == metric_name, :]
            sel_data = metric_df.loc[(metric_df["dimension"] > 1) & (metric_df["dimension"] <= 555), :]
            sel_data["k_groups"] = pd.cut(sel_data["k_val"], k_val_bins, include_lowest=True, right=False)
            sel_data["auc_groups"] = pd.cut(sel_data["auc_score"], auc_bins, include_lowest=True, right=False)

            getCorrelations(sel_data, metric_name, scaling_mode, save_path)
            distanceBoxplot(sel_data, metric_name, scaling_mode, save_path)
            distanceAUCBoxplot(sel_data, metric_name, scaling_mode, save_path)
            distanceDimensionBoxplot(sel_data, metric_name, scaling_mode, save_path)

def allExperiments():
    metrics = ["pr", "dc"]
    scalings = ["real_scaled", "fake_scaled"]
    overview_map_boxplots = "./experiment_figures/boxplot/"
    createPlots(metrics, scalings, overview_map_boxplots)

def smallExperiment():
    metrics = ["pr", "dc"]
    scalings = ["real_scaled"]
    overview_map_boxplots = "./experiment_figures/boxplot/"
    createPlots(metrics, scalings, overview_map_boxplots)

#smallExperiment()
#allExperiments()
