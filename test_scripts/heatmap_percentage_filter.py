import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utility_scripts import helper_functions as util
from utility_scripts import winner_pick_utility as win_util
from pathlib import Path
import time

def heatMap(count_df, y_label_name, sub_map, file_name):
    width = 12
    height = count_df.shape[0]
    #print(count_df.shape)
    plt.figure(figsize=(width, height))
    table_vals = count_df.values
    heatmap = sns.heatmap(table_vals,
                          cmap="RdYlGn_r",
                          xticklabels=count_df.columns,
                          yticklabels=count_df.index,
                          annot=table_vals,
                          annot_kws={"color": "black", "backgroundcolor": "white", "size":9},
                          vmin=0, vmax=100
                          )
    heatmap.invert_yaxis()
    plt.xlabel("k-values")
    plt.xscale("log")
    plt.ylabel(y_label_name)

    Path(sub_map).mkdir(parents=True, exist_ok=True)
    save_path = f"{sub_map}/{file_name}"
    #plt.savefig(save_path, bbox_inches="tight", format="pdf")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

# def plotBoxes(df_results, variable_name, save_path, file_name):
#     plt.figure(figsize=(22, 6))
#     variable_names = sorted(df_results[variable_name].unique())
#     rows = np.ceil(len(variable_names) / 3).astype(int)
#     fig, axs = plt.subplots(rows, 3)
#     axs = axs.flatten()
#     for index, name in enumerate(variable_names):
#         sel_data = df_results.loc[df_results[variable_name] == name, :]
#         k_values = sel_data["k_value"]
#         axs[index].boxplot(k_values)
#         axs[index].set_title(name)
#         #axs[index].set_ylim([1, 999])
#
#     fig.subplots_adjust(wspace=.2, hspace=.4)
#     Path(save_path).mkdir(parents=True, exist_ok=True)
#     save_path = f"{save_path}/{file_name}"
#     plt.savefig(save_path, bbox_inches='tight')
#     plt.close()

def filterKValues(df_top_picks, variable_name):
    variable_names = sorted(df_top_picks[variable_name].unique())
    k_values_included = []
    for var in variable_names:
        sel_data = df_top_picks.loc[df_top_picks[variable_name] == var, :]
        summed_k_counts = sel_data.groupby(["k_value"])["count_value"].sum().reset_index()
        top10_picks = summed_k_counts.nlargest(10, "count_value")
        # Extra check
        k_value_sel = top10_picks["k_value"].unique()
        for k_val in k_value_sel:
            if k_val not in k_values_included:
                k_values_included.append(k_val)

    k_filtered = df_top_picks.loc[df_top_picks["k_value"].isin(k_values_included)]
    return k_filtered

def getPercentageMap(df_top_picks, count_map):
    remaining_results = df_top_picks.loc[df_top_picks["k_value"] >= 1, :]
    percentage_count = remaining_results.groupby(["dimension", "k_value"])["count_value"].sum().unstack()
    total_dim_count = percentage_count.sum(axis=1)
    percentage_count_div = (count_map.div(total_dim_count.values, axis=0) * 100).round(decimals=1)

    return percentage_count_div

def getPercentageMapFilter(df_top_picks, variable_name, percentage_min=10):
    remaining_results = df_top_picks.loc[df_top_picks["k_value"] >= 1, :]
    count_map = remaining_results.groupby([variable_name, "k_value"])["count_value"].sum().unstack()
    total_dim_count = count_map.sum(axis=1)
    percentage_map = (count_map.div(total_dim_count.values, axis=0) * 100)
    included_mask = (percentage_map >= percentage_min).any(axis=0)
    percentage_map_sel = percentage_map.loc[:, included_mask].fillna(0)

    return percentage_map_sel.round(decimals=1)

def saveSmallDF():
    df_read = f"../dataframe_evaluation/real_scaled/combined/dataframe_all.pkl"
    df_save = f"../dataframe_evaluation/test/real_scaled/df_real_test.pkl"
    all_dfs = util.readPickle(df_read)
    dimensions = sorted(all_dfs["dimension"].unique())
    sel_indices = []
    for index, dimension in enumerate(dimensions):
        sel_data = all_dfs.loc[all_dfs["dimension"] == dimension, :]
        sel_data = sel_data.iloc[:999, :]
        indices = sel_data.index
        sel_indices.extend(indices)

    sel_df = all_dfs.loc[all_dfs.index.isin(sel_indices), :]
    util.savePickle(df_save, sel_df)

def getAUCDimMap(dataframe, percentage_min, auc_bins, map_path):
    for auc_index, auc_begin in enumerate(auc_bins[:-1]):
        auc_end = auc_bins[auc_index+1]
        dataframe_top_counts_sel = dataframe.loc[(dataframe["auc_score"] >= auc_begin) & (
                    dataframe["auc_score"] < auc_end), :]
        percentage_count = getPercentageMapFilter(dataframe_top_counts_sel, "dimension", percentage_min=percentage_min)
        file_name = f"auc{auc_begin}-{auc_end}.png"
        heatMap(percentage_count, "Dimension", map_path, file_name)

def getResults(metrics, scalings, max_winner,
               variable_name, map_path, percentage_min):
    auc_bins = [0, 0.1, 0.45, 0.9, 1.1]
    for scaling_mode in scalings:
        df_path_combined = f"../dataframe_evaluation/{scaling_mode}/combined/dataframe_all.pkl"
        #df_path_combined = f"../dataframe_evaluation/test/real_scaled/df_real_test.pkl"
        all_dfs = util.readPickle(df_path_combined)
        for metric_name in metrics:
            metric_df = all_dfs.loc[all_dfs["metric_name"] == metric_name, :]
            dataframe_top_picks = win_util.countTopPicks(metric_df, max_winner, best_mode=True)
            dataframe_top_picks_included = dataframe_top_picks.loc[dataframe_top_picks["k_value"] > -1]
            filtered_data = dataframe_top_picks_included.copy()

            print(dataframe_top_picks.shape)
            print(dataframe_top_picks_included.shape)
            print()
            file_name = f"{metric_name}_{scaling_mode}_{variable_name}.png"
            if "auc" in variable_name:
                filtered_data["auc_groups"] = pd.cut(filtered_data["auc_score"], auc_bins,
                                                           include_lowest=True, right=False)
            if variable_name == "auc_dimension":
                sub_map = f"{map_path}{metric_name}_{scaling_mode}/"
                getAUCDimMap(filtered_data, percentage_min, auc_bins, sub_map)
            else:
                percentage_map_new = getPercentageMapFilter(filtered_data, variable_name,
                                                                percentage_min=percentage_min)
                heatMap(percentage_map_new, variable_name, map_path, file_name)

def smallExperiment():
    #saveSmallDF()
    metrics = ["pr", "dc"]
    scalings = ["real_scaled", "fake_scaled"]
    # 0.002 is max 1 winner
    max_winner = 1
    percentage_min = 0
    map_path = f"./heatmap_percentage_filters/dimension/"
    getResults(metrics, scalings, max_winner, "dimension", map_path, percentage_min)

def allExperiment():
    percentage_min = 10
    metrics = ["pr", "dc"]
    scalings = ["real_scaled", "fake_scaled"]
    # 0.002 is max 1 winner
    max_winner = 1
    map_path = f"./heatmap_percentage_filters/auc_dimension/"
    getResults(metrics, scalings, max_winner, "auc_dimension", map_path, percentage_min)
    map_path = f"./heatmap_percentage_filters/dimension/"
    getResults(metrics, scalings, max_winner, "dimension", map_path, percentage_min)

    map_path = f"./heatmap_percentage_filters/auc_groups/"
    getResults(metrics, scalings, max_winner, "auc_groups", map_path, percentage_min)


smallExperiment()
