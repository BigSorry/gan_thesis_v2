import numpy as np
import helper_functions as util
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.transforms
import pandas as pd
import seaborn as sns
import glob
import auc_dimension_grouping as auc_dim_util

def assignAUCGroup(dataframe, auc_filter):
    dataframe["auc_group"] = -1
    for auc_index, (auc_begin, auc_end) in enumerate(auc_filter):
        bool_array = (dataframe["auc_score"] >= auc_begin) & (dataframe["auc_score"] < auc_end)
        dataframe.loc[bool_array, "auc_group"] = auc_index

def fixSaveDF(dataframe, save_path):
    # Correct (wrong columns columns, old saving)
    actual_distance = dataframe["first_score"]
    actual_first = dataframe["second_score"]
    actual_second = dataframe["distance"]
    dataframe["distance"] = actual_distance
    dataframe["first_score"] = actual_first
    dataframe["second_score"] = actual_second
    dataframe.to_pickle(f"{save_path}dataframe_all.pkl")

def filterGroupedData(group, best_mode):
    if best_mode:
        top_distance = group.nsmallest(1, "distance").loc[:, "distance"].min()
        boolean_filter = (group["distance"] <= top_distance)# | (
            #np.isclose(group["distance"], [top_distance], atol=1e-2))
    else:
        top_distance = group.nlargest(1, "distance").loc[:, "distance"].max()
        boolean_filter = (group["distance"] >= top_distance)# | (
            #np.isclose(group["distance"], [top_distance], atol=1e-2))
    filter_data = group.loc[boolean_filter, :]

    return filter_data, top_distance

def debugFiltered(group_data, title_text):
    plt.figure()
    plt.title(title_text)
    first_score = group_data["first_score"]
    second_score = group_data["second_score"]
    plt.scatter(second_score, first_score, color="green")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()


def overiewBoxplot(dataframe, metric_name, scaling_mode, best_mode, max_filter, save_map):
    dim_data = dataframe.loc[(dataframe["dimension"] > 1) & (dataframe["dimension"] <= 555), :]
    sorted_auc = sorted(dim_data["auc_group"].unique())
    sorted_dims = sorted(dim_data["dimension"].unique())
    k_values = {auc : [] for auc in sorted_auc}
    good_runs = 0
    fail_runs = 0
    dimension_fails = {dim:0 for dim in sorted_dims}
    for auc_index in sorted_auc:
        sel_data = dim_data.loc[dim_data["auc_group"] == auc_index, :]
        grouped_data = sel_data.groupby(["iter", "auc_score", "dimension"])

        for name, group in grouped_data:
            filter_data, top_distance = filterGroupedData(group, best_mode)
            filter_percentage = filter_data.shape[0] / group.shape[0]

            if filter_percentage < max_filter:
                best_picks = filter_data["k_val"].values
                k_values[auc_index].extend(best_picks)
            else:
                fail_runs+=1
                dimension = name[2]
                dimension_fails[dimension] += 1
                #debugFiltered(group, name)

    print(dimension_fails)
    print()

    auc_k_values = list(k_values.values())
    # Global variable
    x_new_ticks = [f"{begin}-{min(1 , end)}" for (begin, end) in auc_filter]
    plt.boxplot(auc_k_values)
    plt.ylabel("K-value")
    plt.yscale("log")
    plt.xticks(np.arange(len(x_new_ticks)) + 1, x_new_ticks)
    plt.xlabel("AUC range")
    best_mode_str = "best_pick" if best_mode else "worst_pick"
    sub_map = f"{save_map}/{best_mode_str}/f{experiment_filter}/"
    Path(sub_map).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{sub_map}{metric_name}_{scaling_mode}.png", bbox_inches="tight")
    plt.close()

def createPlots(metrics, scalings, auc_filter, experiment_filter, path_box, best_mode=True):
    Path(path_box).mkdir(parents=True, exist_ok=True)
    read_all = False
    for scaling_mode in scalings:
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
            overiewBoxplot(metric_df, metric_name, scaling_mode, best_mode, experiment_filter, path_box)


metrics = ["dc", "pr"]
scalings = ["real_scaled", "fake_scaled"]
#scalings = ["real_scaled"]
overview_map_boxplots = "./gaussian_dimension/no_dim_group/"
auc_filter = [(0, 0.3), (0.3, 0.7), (0.7, 1.1)]
experiment_filters = [0.1, 0.5, 1.1]

for experiment_filter in experiment_filters:
    createPlots(metrics, scalings, auc_filter, experiment_filter, overview_map_boxplots, best_mode=True)
    createPlots(metrics, scalings, auc_filter, experiment_filter, overview_map_boxplots, best_mode=False)
