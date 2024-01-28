import numpy as np
from utility_scripts import helper_functions as util
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from test_scripts.boxplot_all_k import createBoxplot, createLines

def plotNoGroupLines(dataframe, quantile_factor, label_name):
    quantile = int(quantile_factor * 100)
    plt.title(f"Quantile {quantile}th distance")
    quantile_distance = dataframe.groupby("k_val")["distance"].quantile(q=quantile_factor)
    x_vals = np.arange(quantile_distance.shape[0]) + 1
    plt.plot(x_vals, quantile_distance, label=label_name)
    plt.xlabel("K-value")
    plt.ylabel("Distance")
    plt.ylim([0, 1.1])
    plt.xscale("log")

def plotBoxes(dataframe, metric_name, parent_map):
    boxes = dataframe.groupby("k_val").apply(lambda x: [x["distance"]])
    max_value = dataframe["distance"].max()
    print(max_value)
    createBoxplot("", boxes, max_value)

def getFilterMask(dataframe, metric_name, scaling_mode, filter_percentage=0.002):
    filter_mask = np.ones(dataframe.shape[0]).astype(bool)
    excluded_keys_dict = util.readPickle(f"../filter_keys/{metric_name}_{scaling_mode}.pkl")
    excluded_keys = excluded_keys_dict[filter_percentage]
    for (iter, dimension, auc_score) in excluded_keys:
        excluded = (dataframe["iter"] == iter) & (dataframe["dimension"] == dimension) & (dataframe["auc_score"] == auc_score)
        filter_mask[excluded] = False

    return filter_mask

def getAllDataframes(metrics, scalings, max_dim, filter_percentage):
    dict_dfs = {}
    for scaling_mode in scalings:
        df_path_combined = f"../dataframe_evaluation/{scaling_mode}/combined/dataframe_all.pkl"
        all_dfs = util.readPickle(df_path_combined)
        for metric_name in metrics:
            metric_df = all_dfs.loc[all_dfs["metric_name"] == metric_name, :]
            sel_data = metric_df.loc[(metric_df["dimension"] > 1) & (metric_df["dimension"] <= max_dim), :]
            filter_mask = getFilterMask(sel_data, metric_name, scaling_mode, filter_percentage=filter_percentage)
            filtered_data = sel_data.iloc[filter_mask, :].copy()
            if metric_name not in dict_dfs:
                dict_dfs[metric_name] = filtered_data
            else:
                dict_dfs[metric_name] = pd.concat([dict_dfs[metric_name], filtered_data], ignore_index=True)

    return dict_dfs

def plotMetrics(metrics, scalings, filter_percentage, max_dim, parent_map):
    # save sub map if needed
    plt.figure()
    df_merged = getAllDataframes(metrics, scalings, max_dim, filter_percentage)
    for metric_name in metrics:
        sel_data = df_merged[metric_name]
        maximum_distance = sel_data.groupby("k_val")["distance"].quantile(q=0.75)
        minimum_distance = sel_data.groupby("k_val")["distance"].quantile(q=0.25)
        distance_median = sel_data.groupby("k_val")["distance"].quantile(q=0.5)
        plotBoxes(sel_data, metric_name, parent_map)
        createLines(distance_median, minimum_distance, maximum_distance)

        Path(parent_map).mkdir(parents=True, exist_ok=True)
        save__path = f"{parent_map}{metric_name}_boxplot.pdf"
        plt.savefig(save__path, bbox_inches="tight")
        plt.close()

def plotScenarios(metrics, scalings, filter_percentage, quantiles, max_dim, parent_map):
    # save sub map if needed
    for quant in quantiles:
        plt.figure()
        for scaling_mode in scalings:
            df_path_combined = f"../dataframe_evaluation/{scaling_mode}/combined/dataframe_all.pkl"
            all_dfs = util.readPickle(df_path_combined)
            for metric_name in metrics:
                metric_df = all_dfs.loc[all_dfs["metric_name"] == metric_name, :]
                sel_data = metric_df.loc[(metric_df["dimension"] > 1) & (metric_df["dimension"] <= max_dim), :]
                filter_mask = getFilterMask(sel_data, metric_name, scaling_mode, filter_percentage=filter_percentage)
                filtered_data = sel_data.iloc[filter_mask, :].copy()

                label_name = f"{metric_name}_{scaling_mode}"
                plotNoGroupLines(filtered_data, quant, label_name)

        plt.legend(fontsize="x-small", ncol=2)
        Path(parent_map).mkdir(parents=True, exist_ok=True)
        save__path = f"{parent_map}q{quant}.pdf"
        plt.savefig(save__path, bbox_inches="tight")
        plt.close()

def allExperiments():
    metrics = ["pr", "dc"]
    scalings = ["real_scaled", "fake_scaled"]
    # Max 1 winner
    filter_percentages = [0.002]
    quantiles = [0.1, 0.25, 0.5, 0.75]
    max_dim = 555
    for filter_percentage in filter_percentages:
        parent_map = f"../experiment_figures/lines/no_group/"
        plotScenarios(metrics, scalings, filter_percentage, quantiles, max_dim, parent_map)
        parent_map = f"../experiment_figures/lines/no_group/metrics_only/"
        plotMetrics(metrics, scalings, filter_percentage, max_dim, parent_map)

def smallExperiment():
    metrics = ["pr", "dc"]
    scalings = ["real_scaled", "fake_scaled"]
    filter_percentage = 0.002
    max_dim = 555
    quantiles = [0.75]
    parent_map = f"../experiment_figures/lines/no_group/"
    #plotScenarios(metrics, scalings, filter_percentage, quantiles, max_dim, parent_map)
    parent_map = f"../experiment_figures/lines/no_group/metrics_only/"
    plotMetrics(metrics, scalings, filter_percentage, max_dim, parent_map)

allExperiments()
