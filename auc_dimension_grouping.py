import numpy as np
import helper_functions as util
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.transforms
import pandas as pd
import seaborn as sns
import glob

def expectedK(k_values, counts):
    counts_np = np.array(counts)
    counts_np[counts_np <= 5] = 0
    densities_normed = counts_np / np.sum(counts_np)
    contributions = k_values * densities_normed
    expected_value = np.sum(contributions)

    return expected_value

def getExpectedValueDict(overview_count_plot):
    plot_info = {}
    for (metric_name, scaling_mode), auc_dict in overview_count_plot.items():
        for auc_index, dimension_dict in auc_dict.items():
            for dimension, count_dict in dimension_dict.items():
                plot_key = (auc_index, dimension)
                if plot_key not in plot_info:
                    plot_info[plot_key]= {}
                k_vals = list(count_dict.keys())
                counts = list(count_dict.values())
                expected_value_top_k = expectedK(k_vals, counts)
                plot_info[plot_key] = expected_value_top_k

    return plot_info

def overviewPlot(plot_dict, metric_name, scaling_mode, save_map):
    key_pairs = np.array(list(plot_dict.keys()))
    sorted_dims = sorted(np.unique(key_pairs[:, 1]))
    sorted_auc = sorted(np.unique(key_pairs[:, 0]))
    for auc_index in sorted_auc:
        x_vals = []
        y_vals = []
        for dimension_index, dimension in enumerate(sorted_dims):
            expected_value = np.round(plot_dict[(auc_index, dimension)], 2)
            x_vals.append(dimension)
            y_vals.append(expected_value)
        plt.plot(x_vals, y_vals, label=auc_index)
    plt.legend()
    plt.savefig(f"{save_map}overview_plot_{metric_name}_{scaling_mode}.png", bbox_inches="tight")
    plt.close()

def changeTableCellColor(cell_values, table, row_id, compare_val):
    for column_id in range(cell_values.shape[1]):
        value = cell_values[row_id, column_id]
        if value >= compare_val:
            table._cells[(row_id, column_id)]._text.set_color("white")
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
    grouped_data = dimension_data.groupby(["iter", "auc_score"])
    k_values = []
    avg_picks = 0
    index = 0
    avg_distance = 0
    for name, group in grouped_data:
        filter_data, top_distance = filterGroupedData(group, best_mode)
        best_picks = filter_data["k_val"].values

        avg_picks += best_picks.shape[0]
        avg_distance += top_distance
        # Short fix grouping same auc score within same iteration
        within_groups = np.ceil(group.shape[0] / 999)
        index+=within_groups
        filter_percentage = filter_data.shape[0] / group.shape[0]
        if filter_percentage < .1:
            k_values.extend(best_picks)


    avg_picks = np.round(avg_picks / index, 1)
    avg_distance = np.round(avg_distance / index, 1)

    return k_values, avg_picks, index, avg_distance

def createSmallTable(legend_info, dimensions):
    xmin, ymin, width, height = 0, -0.65, 1, 0.5
    table_vals = np.array(legend_info).T
    colors = np.full(table_vals.shape, "w")
    colors[0, :] = "000000"
    colors[2, table_vals[2, :] >= 10] = "000000"
    table = plt.table(cellText=table_vals,
              rowLabels=["Dimension", "Total groups", "Average k-vals pick", "avg_distance"],
              cellLoc="center",
              cellColours=colors,
              bbox=(xmin, ymin, width, height))

    changeTableCellColor(table_vals, table, 0, 2)
    changeTableCellColor(table_vals, table, 2, 10)
    table.set_fontsize(16)

def overviewBoxplot(dataframe, metric_name, scaling_mode, best_mode, save_map):
    sel_data = dataframe.loc[(dataframe["dimension"] > 1) & (dataframe["dimension"] <= 555), :]
    dimensions_sorted = sorted(sel_data["dimension"].unique())
    sorted_auc = sorted(sel_data["auc_group"].unique())
    for auc_index in sorted_auc:
        k_picks = []
        label_info = []
        legend_info = []
        sel_data = dataframe.loc[dataframe["auc_group"] == auc_index, :]
        if sel_data.shape[0] > 0:
            for dimension in dimensions_sorted:
                dimension_data = sel_data.loc[sel_data["dimension"] == dimension]
                best_picks, avg_picks, groups, avg_distance = getBestValues(dimension_data, best_mode=best_mode)
                k_picks.append(best_picks)
                label_info.append(f"{dimension}")
                legend_info.append([dimension, groups, avg_picks, avg_distance])#filter_data.shape[0], np.round(top_distance, 2)])

            #new_y = makeLabel(label_info)
            old_y = np.arange(len(dimensions_sorted)) + 1
            plt.boxplot(k_picks, vert=False)
            createSmallTable(legend_info, dimensions_sorted)
            plt.yticks(old_y, label_info)
            plt.ylabel("Dimension")
            plt.xscale("symlog")
            plt.xlim([0, 1000])
            plt.xlabel("K-value (log scale)")

            best_mode_str = "best_pick" if best_mode else "worst_pick"
            sub_map = f"{save_map}/{best_mode_str}/"
            Path(sub_map).mkdir(parents=True, exist_ok=True)
            plt.savefig(f"{sub_map}{metric_name}_{scaling_mode}_auc{auc_index}.png", bbox_inches="tight")
            plt.close()

def overviewTablePlot(plot_dict, metric_name, scaling_mode, save_map):
    key_pairs = np.array(list(plot_dict.keys()))
    sorted_dims = sorted(np.unique(key_pairs[:, 1]))
    sorted_auc = sorted(np.unique(key_pairs[:, 0]))
    auc_labels = [f"auc_{auc} " for auc in sorted_auc]
    table_array = np.zeros((len(sorted_auc), len(sorted_dims)))
    cell_colors = np.zeros((len(sorted_auc), len(sorted_dims))).astype(str)
    for auc_index in sorted_auc:
        for dimension_index, dimension in enumerate(sorted_dims):
            expected_value = np.round(plot_dict[(auc_index, dimension)], 2)
            table_array[auc_index][dimension_index] = expected_value
            color = "red" if expected_value > 100 else "w"
            cell_colors[auc_index][dimension_index] = color

    table = plt.table(cellText=table_array,
                      cellColours=cell_colors,
                      rowLabels=auc_labels,
                      colLabels=sorted_dims,
                      loc='center')
    plt.axis('off')
    plt.axis('off')
    plt.gcf().canvas.draw()
    # get bounding box of table
    points = table.get_window_extent(plt.gcf()._cachedRenderer).get_points()
    # add 10 pixel spacing
    points[0, :] -= 10
    points[1, :] += 10
    # get new bounding box in inches
    nbbox = matplotlib.transforms.Bbox.from_extents(points / plt.gcf().dpi)
    # save and clip by new bounding box
    plt.savefig(f"{save_map}overview_{metric_name}_{scaling_mode}.png", bbox_inches=nbbox)
    plt.close()

def overiewBarPlot(plot_dict, metric_name, scaling_mode, save_map):
    key_pairs = np.array(list(plot_dict.keys()))
    sorted_auc = sorted(np.unique(key_pairs[:, 0]))
    for auc in sorted_auc:
        x_vals = []
        y_labels = []
        for (other_auc, dimension), expected_value in plot_dict.items():
            if auc == other_auc:
                x_vals.append(expected_value)
                y_labels.append(dimension)

        y_vals = np.arange(len(y_labels)) + 1
        plt.barh(y_vals, x_vals)
        plt.yticks(y_vals, y_labels)
        max_expected_value = np.max(x_vals)
        if max_expected_value < 21:
            plt.xlim([0, 20])
        plt.savefig(f"{save_map}overview_{metric_name}_{scaling_mode}_auc{auc}.png", bbox_inches='tight')
        plt.close()

def getOverview(filtered_df, auc_filter):
    sorted_dims = sorted(filtered_df["dimension"].unique())
    score_dict = {i:{} for i in range(len(auc_filter))}
    for auc_index, (auc_begin, auc_end) in enumerate(auc_filter):
        sel_dataframe = filtered_df.loc[(filtered_df["auc_score"] >= auc_begin) & (filtered_df["auc_score"] <= auc_end), :]
        for index, dim in enumerate(sorted_dims):
            if dim not in score_dict[auc_index]:
                score_dict[auc_index][dim] = {}
            dimension_data = sel_dataframe.loc[sel_dataframe["dimension"] == dim, :]
            if dimension_data.shape[0] > 0:
                filter_data = dimension_data.groupby(["iter", "auc_score"]).\
                    apply(lambda x: x.nsmallest(n=1, columns='distance')).reset_index(drop=True)
                counted_data = filter_data.groupby("k_val").size()
                for k_val, counts in counted_data.items():
                    if k_val not in score_dict[auc_index][dim]:
                        score_dict[auc_index][dim][k_val] = counts
                    else:
                        score_dict[auc_index][dim][k_val] += counts
    return score_dict

def resampleDF(dataframe, dimensions, min_group_count):
    taken_indices = []
    for dimension in dimensions:
        dimension_data = dataframe.loc[dataframe["dimension"] == dimension, :]
        df_elements = dimension_data.sample(n=min_group_count)
        taken_indices.extend(list(df_elements.index))

    return taken_indices
def aucDimensionOverlap(dataframe, auc_filter):
    dimensions = dataframe["dimension"].unique()
    all_indices = []
    for auc_index, (auc_begin, auc_end) in enumerate(auc_filter):
        sel_data = dataframe.loc[(dataframe["auc_group"] == auc_index), :]
        group_count = sel_data.groupby("dimension").size()
        smallest_group = group_count.min()
        taken_indices = resampleDF(sel_data, dimensions, smallest_group)
        all_indices.extend(taken_indices)

    end_dataframe = dataframe.loc[dataframe.index.isin(all_indices), :]
    return end_dataframe

def assignAUCGroup(dataframe, auc_filter):
    dataframe["auc_group"] = -1
    for auc_index, (auc_begin, auc_end) in enumerate(auc_filter):
        bool_array = (dataframe["auc_score"] >= auc_begin) & (dataframe["auc_score"] < auc_end)
        dataframe.loc[bool_array, "auc_group"] = auc_index
def aucFairness(dataframe):
    dimensions = dataframe["dimension"].unique()
    auc_scores = dataframe["auc_score"].unique()
    smallest_group = -1
    all_indices = []
    for dimension in dimensions:
        dimension_data = dataframe.loc[dataframe["dimension"] == dimension, :]
        group_count = dimension_data.groupby("auc_score").size()
        smallest_group = group_count.min()
        for auc_index, auc_score in enumerate(auc_scores):
            dimension_auc_data = dimension_data.loc[dimension_data["auc_score"] == auc_score, :]
            df_elements = dimension_auc_data.iloc[:999, :]
            all_indices.extend(list(df_elements.index))

    end_dataframe = dataframe.loc[dataframe.index.isin(all_indices), :]
    return end_dataframe

def fixSaveDF(dataframe, save_path):
    # Correct (wrong columns columns, old saving)
    actual_distance = dataframe["first_score"]
    actual_first = dataframe["second_score"]
    actual_second = dataframe["distance"]
    dataframe["distance"] = actual_distance
    dataframe["first_score"] = actual_first
    dataframe["second_score"] = actual_second
    dataframe.to_pickle(f"{save_path}dataframe_all.pkl")


def createPlots(metrics, scalings, auc_filter, path_box, best_mode=True):
    Path(path_box).mkdir(parents=True, exist_ok=True)
    overview_dict = {}
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
metrics = ["pr", "dc"]
scalings = ["real_scaled", "fake_scaled"]
overview_map_boxplots = "./gaussian_dimension/boxplots/"
auc_filter = [(0, 0.3), (0.3, 0.7), (0.7, 1.1)]
createPlots(metrics, scalings, auc_filter, overview_map_boxplots, best_mode=True)
#createPlots(metrics, scalings, auc_filter, overview_map_boxplots, best_mode=False)

