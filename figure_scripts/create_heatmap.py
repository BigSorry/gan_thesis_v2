import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib.colors import LogNorm
from utility_scripts import winner_pick_utility as win_util
from utility_scripts import helper_functions as util

# Split old dfs and save
def saveDFs():
    metrics = ["pr", "dc"]
    scaling_mode = ["real_scaled", "fake_scaled"]
    for scaling in scaling_mode:
        df_path = f"../dataframe_evaluation/{scaling}/combined/dataframe_all.pkl"
        df_scaling = util.readPickle(df_path)
        for metric_name in metrics:
            metric_df = df_scaling.loc[df_scaling["metric_name"] == metric_name, :]
            df_save = f"../dataframe_evaluation/{scaling}/combined/dataframe_{metric_name}.pkl"
            metric_df.to_pickle(df_save)

# Not needed here but method is only here.
def tablePrep(table, save_path):
    # prepare for saving:
    # draw canvas once
    plt.gcf().canvas.draw()
    # get bounding box of table
    points = table.get_window_extent(plt.gcf()._cachedRenderer).get_points()
    # add 10 pixel spacing for borders
    points[0, :] -= 10
    points[1, :] += 10
    # get new bounding box in inches
    nbbox = matplotlib.transforms.Bbox.from_extents(points / plt.gcf().dpi)
    # save and clip by new bounding box
    Path(save_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches=nbbox)
    plt.close()

def prepKvalueTicks(values):
    result = []
    for interval in values:
        range = interval.length
        if range == 1:
            result.append(interval.left)
        else:
            result.append(f"{interval.left}-{interval.right-1}")

    return result

def prepAUCTicks(values):
    result = []
    for interval in values:
        end = interval.right
        if end < 1:
            y_tick = f"[{interval.left}-{interval.right})"
        else:
            y_tick = f"[{interval.left}-1]"
        result.append(y_tick)

    return result

def heatMap(count_df, y_label_name, sub_map, file_name):
    width = 8
    width = 14
    height = count_df.index.shape[0]
    plt.figure(figsize=(width, height))
    table_vals = count_df.values
    k_value_ticks = prepKvalueTicks(count_df.columns)
    y_ticks = count_df.index
    if y_label_name == "AUC Range":
        y_ticks = prepAUCTicks(count_df.index)
    heatmap = sns.heatmap(table_vals,
                          cmap="RdYlGn_r",
                          xticklabels=k_value_ticks,
                          yticklabels=y_ticks,
                          annot=table_vals,
                          annot_kws={"color": "black", "backgroundcolor": "white", "size":9},
                          vmin=0, vmax=100
                          )
    heatmap.invert_yaxis()
    plt.xlabel("K-value range")
    plt.ylabel(y_label_name)

    Path(sub_map).mkdir(parents=True, exist_ok=True)
    save_path = f"{sub_map}/{file_name}"
    plt.savefig(save_path+".pdf", bbox_inches="tight", format="pdf", dpi=150)
    plt.savefig(save_path+".png", bbox_inches="tight", dpi=150)
    plt.close()

def getPercentageMap(dataframe_top_counts, first_grouping):
    # Get amount of experiments in a certain auc range
    count_map = dataframe_top_counts.groupby([first_grouping, "k_groups"])["count_value"].sum().unstack()
    # Testing which k_vals has high percentage
    # threshold = 5
    # mask_high = count_map > threshold
    # df_high = count_map[mask_high]
    # df_cleaned = df_high.dropna(axis=1, how='all')
    # k_vals = list(df_cleaned)
    # print(k_vals)
    total_count = count_map.sum(axis=1)
    percentage_count_div = (count_map.div(total_count.values, axis=0) * 100).round(decimals=2)

    return percentage_count_div

def saveAUC(dataframe_top_picks, parent_path, file_name):
    sub_map = f"{parent_path}/auc/"
    Path(sub_map).mkdir(parents=True, exist_ok=True)
    # save sub map if needed
    percentage_count = getPercentageMap(dataframe_top_picks, "auc_groups")
    # Percentage method
    heatMap(percentage_count, "AUC Range", sub_map, file_name)

def saveDimension(dataframe_top_picks, parent_path, file_name):
    # All auc for dimensions
    sub_map = f"{parent_path}/dimension/"
    Path(sub_map).mkdir(parents=True, exist_ok=True)

    percentage_count = getPercentageMap(dataframe_top_picks, "dimension")
    heatMap(percentage_count, "Dimension", sub_map, file_name)

def saveAUCDimension(dataframe_top_picks, auc_bins, map_path, scenario_str):
    sub_map = f"{map_path}/auc_dimension/{scenario_str}"
    Path(sub_map).mkdir(parents=True, exist_ok=True)
    for auc_index, auc_end in enumerate(auc_bins[1:]):
        auc_begin = auc_bins[auc_index]
        df_sel = dataframe_top_picks.loc[(dataframe_top_picks["auc_score"] >= auc_begin) & (
                    dataframe_top_picks["auc_score"] < auc_end), :]
        auc_end_str = f"{auc_end})"
        if auc_end > 1:
            auc_end_str = "1]"
        file_name = f"auc[{auc_begin}-{auc_end_str}.png"
        percentage_count = getPercentageMap(df_sel, "dimension")
        heatMap(percentage_count, "Dimension", sub_map, file_name)

def checkAUCSpread(dataframe, scenario_str):
    dimensions = dataframe["dimension"].unique()
    map_path = f"../experiment_figures/auc_spread_winners/{scenario_str}/"
    Path(map_path).mkdir(parents=True, exist_ok=True)
    for dim in dimensions:
        plt.title(f"Dimension {dim}")
        dimension_data = dataframe.loc[dataframe["dimension"] == dim, :]
        dimension_data["auc_score"].plot.hist(bins=3, edgecolor='black')
        save_path = f"{map_path}dimension{dim}.png"
        plt.savefig(save_path)
        plt.close()

def createPlots(scaling_modes, metrics, max_winner, max_dimension, save_path, best_mode=True):
    # save sub map if needed
    best_str = "best" if best_mode else "worst"
    parent_path = f"{save_path}max_winner{max_winner}_pick_{best_str}/"
    #auc_bins = [0, 0.1, 0.5, 0.9, 1.1]
    #auc_bins = [0, 0.25, 0.5, 0.75, 1.1]
    auc_bins = [0, 0.3, 0.5, 0.9, 1.1]
    auc_bins = [0, .3, .7, 1.1]
    for scaling_mode in scaling_modes:
        for metric_name in metrics:
            scenario_str = f"{metric_name}_{scaling_mode}"
            print(scenario_str)
            df_path = f"../dataframe_evaluation/{scaling_mode}/combined/dataframe_{metric_name}.pkl"
            df = util.readPickle(df_path)
            sel_data = df.loc[(df["dimension"] > 1) & (df["dimension"] <= max_dimension), :]

            dataframe_top_picks = win_util.countTopPicks(sel_data, max_winner, best_mode)
            dataframe_top_picks_included = dataframe_top_picks.loc[dataframe_top_picks["k_value"] > -1]
            filtered_data = dataframe_top_picks_included.copy()
            makeKvalGroups(filtered_data)
            filtered_data["auc_groups"] = pd.cut(filtered_data["auc_score"], auc_bins, include_lowest=True, right=False)
            print(auc_bins)
            # Plotting but grouping results differently
            #checkAUCSpread(filtered_data, scenario_str)

            saveAUC(filtered_data, parent_path, file_name=f"{scenario_str}_auc")
            saveDimension(filtered_data, parent_path, file_name=f"{scenario_str}_dimension")
            saveAUCDimension(filtered_data, auc_bins, parent_path, scenario_str)

def makeKvalGroups(df):
    first_intervals = [i for i in range(1, 12)]
    middle_interval = [51, 501]
    end_interval = [999, 1000]

    bins = first_intervals + middle_interval + end_interval
    df["k_groups"] = pd.cut(df["k_value"], bins, include_lowest=True, right=False)

def allExperiments():
    metrics = ["pr", "dc"]
    scaling_modes = ["real_scaled", "fake_scaled"]
    heatmap_path = "../experiment_figures/heatmap_test/"
    max_winner = 1
    max_dimension = 555
    createPlots(scaling_modes, metrics, max_winner, max_dimension, heatmap_path, best_mode=True)

def smallExperiment():
    metrics = ["dc"]
    scaling_modes = ["real_scaled"]
    heatmap_path = "../experiment_figures/heatmap_test/"
    max_winner = 1
    max_dimension = 555
    createPlots(scaling_modes, metrics, max_winner, max_dimension, heatmap_path, best_mode=True)

allExperiments()
#smallExperiment()