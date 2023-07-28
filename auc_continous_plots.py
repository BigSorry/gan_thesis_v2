import numpy as np
import helper_functions as util
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.transforms
import pandas as pd
import seaborn as sns
import glob
from matplotlib import cm

def filterGroupedData(group, best_mode):
    if best_mode:
        top_distance = group.nsmallest(1, "distance").loc[:, "distance"].max()
        boolean_filter = (group["distance"] <= top_distance) #| (
            #np.isclose(group["distance"], [top_distance], atol=1e-2))
    else:
        top_distance = group.nlargest(1, "distance").loc[:, "distance"].max()
        boolean_filter = (group["distance"] >= top_distance) #| (
            #np.isclose(group["distance"], [top_distance], atol=1e-2))
    filter_data = group.loc[boolean_filter, :]

    return filter_data, top_distance

def getBestValues(dimension_data, best_mode=True):
    grouped_data = dimension_data.groupby(["iter", "dimension", "auc_score"])
    auc_dict = {}
    for name, group in grouped_data:
        filter_data, top_distance = filterGroupedData(group, best_mode)
        best_picks = filter_data["k_val"].values
        auc_scores = filter_data["auc_score"].values
        filter_percentage = filter_data.shape[0] / group.shape[0]
        if filter_percentage < .1:
            for i, k_val in enumerate(best_picks):
                auc_score = auc_scores[i]
                if k_val not in auc_dict:
                    auc_dict[k_val] = [auc_score]
                else:
                    auc_dict[k_val].append(auc_score)

    return auc_dict

def testHeatmap(count_map, xticks, yticks):
    # Heatmap
    plt.figure()
    ax = sns.heatmap(count_map,
                     cmap="RdYlGn_r",
                     xticklabels=xticks,
                     yticklabels=yticks,
                     annot=count_map,
                     annot_kws={"color": "black", "backgroundcolor": "white"},
                     vmin=0,
                     )
    ax.invert_yaxis()
    plt.yticks(rotation=45)

def plotBox(dict_results, k_boundaries):
    auc_scores_list = list(dict_results.values())
    bins = 4
    array_bins, step = np.linspace(0, 1, bins+1, retstep=True)
    array_bins = array_bins.round(decimals=2)
    print(array_bins)
    count_map = np.zeros((bins, len(auc_scores_list)))
    for column_index, score_list in enumerate(auc_scores_list):
        auc_values = pd.Series(score_list)
        auc_values[auc_values >= 1] = 0.99
        auc_values[auc_values == 0] = 0.01
        auc_categorical = pd.cut(auc_values, array_bins)
        counted = auc_categorical.value_counts()
        counts = counted.values
        decimal_labels = list(counted.index)
        for index, count in enumerate(counts):
            begin_interval = decimal_labels[index].left
            # This can be risky with decimal comparison
            row_index = np.where(array_bins == begin_interval)[0]
            count_map[row_index, column_index] = int(count)

    # Bars
    plt.figure()
    print(count_map)
    auc_labels = [f"({array_bins[i]}, {array_bins[i+1]}]" for i in range(len(array_bins)-1)]
    auc_labels[0] = auc_labels[0].replace("(", "[")
    # Set position of bar on X axis
    bar_width = 0.2
    start_x = np.arange(count_map.shape[1]) - ((bins // 2) - bar_width)

    for i in range(count_map.shape[0]):
        new_x = start_x + (bar_width*i)
        plt.bar(new_x, count_map[i, :], width=bar_width, label=auc_labels[i])

    xticks = [f"k-values {begin}-{end}" for (begin, end) in k_boundaries]
    #old_tick = [x + bar_width for x in start_x]
    plt.xticks(start_x, xticks, rotation=90)
    plt.yscale("log")
    plt.ylabel("Count")
    max_y = np.max(count_map) * 10
    plt.ylim([1, max_y])
    plt.legend()


def overviewBoxplot(dataframe, metric_name, scaling_mode, best_mode, save_map):
    sel_data = dataframe.loc[(dataframe["dimension"] > 1) & (dataframe["dimension"] <= 555), :]
    auc_dict = getBestValues(sel_data, best_mode=best_mode)
    k_values = list(auc_dict.keys())
    group_step = 500
    k_groups = [(i-group_step, i) for i in range(group_step, 1001, group_step)]
    #k_groups = [(0, 1001)]
    step_size = 50
    print(k_groups)
    for i, (begin_k, end_k) in enumerate(k_groups):
        new_dict = {}
        k_boundaries = [(i, i+step_size) for i in range(begin_k, end_k, step_size)]
        print(k_boundaries)
        print()
        for i, (begin, end) in enumerate(k_boundaries):
            new_dict[(i+1)] = []
            for k_value in k_values:
                if k_value >= begin and k_value < end:
                    auc_scores = auc_dict[k_value]
                    new_dict[(i+1)].extend(auc_scores)

        k_value_groups = list(new_dict.keys())
        auc_scores_list = list(new_dict.values())
        plotBox(new_dict, k_boundaries)

        # Scatter
        # plt.figure()
        # for i, auc_scores in enumerate(auc_scores_list):
        #     y_vals = auc_scores
        #     k_value_group = k_value_groups[i]
        #     x_vals = [k_value_group for i in range(len(y_vals))]
        #     plt.scatter(x_vals, y_vals)
        #
        #
        #
        # plt.ylabel("Auc score")
        # plt.ylim([0, 1.1])
        # plt.xlabel("K-value group")
        # plt.xticks(np.arange(len(k_value_groups)) + 1, k_value_groups)
        best_mode_str = "best_pick" if best_mode else "worst_pick"
        sub_map = f"{save_map}/{best_mode_str}/"
        Path(sub_map).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{sub_map}{metric_name}_{scaling_mode}_{begin_k}-{end_k}.png", bbox_inches="tight")
        plt.close()

def assignAUCGroup(dataframe, auc_filter):
    dataframe["auc_group"] = -1
    for auc_index, (auc_begin, auc_end) in enumerate(auc_filter):
        bool_array = (dataframe["auc_score"] >= auc_begin) & (dataframe["auc_score"] < auc_end)
        dataframe.loc[bool_array, "auc_group"] = auc_index

def createPlots(metrics, scalings, auc_filter, path_box, best_mode=True):
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
            overviewBoxplot(metric_df, metric_name, scaling_mode, best_mode, path_box)


metrics = ["pr", "dc"]
metrics = ["pr"]
scalings = ["real_scaled", "fake_scaled"]
scalings = ["real_scaled"]
overview_map_boxplots = "./gaussian_dimension/scatter_auc/"
auc_filter = [(0, 0.3), (0.3, 0.7), (0.7, 1.1)]
auc_filter = [(0, 1.1)]
createPlots(metrics, scalings, auc_filter, overview_map_boxplots, best_mode=True)
#createPlots(metrics, scalings, auc_filter, overview_map_boxplots, best_mode=False)
plt.show()
