import numpy as np
import helper_functions as util
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.transforms
import pandas as pd
import seaborn as sns
def dfDistancePlots(dataframe, auc_filter, sub_map):
    sorted_dims = sorted(dataframe["dimension"].unique())
    for auc_index, (auc_begin, auc_end) in enumerate(auc_filter):
        sel_dataframe = dataframe.loc[(dataframe["auc_score"] >= auc_begin) & (dataframe["auc_score"] <= auc_end), :]
        auc_map = f"{sub_map}/auc{auc_index}/"
        Path(auc_map).mkdir(parents=True, exist_ok=True)
        for index, dim in enumerate(sorted_dims):
            dimension_data = sel_dataframe.loc[sel_dataframe["dimension"] == dim, :]
            if dimension_data.shape[0] > 0:
                plt.figure(figsize=(16, 8))
                filter_data = dimension_data.groupby(["iter", "auc_score"]).\
                    apply(lambda x: x.nsmallest(n=5, columns='distance')).reset_index(drop=True)
                grouped_data = filter_data.groupby("k_val")
                counted_data = grouped_data.size()
                counted_data = counted_data[counted_data > 5]
                k_vals = counted_data.index.values
                x = np.arange(k_vals.shape[0]) + 1
                y = counted_data.values
                plt.subplot(2, 1, 1)
                plt.bar(x, y)
                plt.xticks(x, k_vals)
                plt.xlabel("K-values")
                if k_vals.shape[0] > 50:
                    plt.xscale("log")
                # Boxplots
                groups = dimension_data.groupby("k_val")
                x = list(groups.groups.keys())
                y = groups['distance'].apply(np.hstack).values
                plt.subplot(2,1,2)
                plt.boxplot(y.T, positions=x)
                plt.xscale("log")

                plt.ylabel("Count Top-10 per auc score")
                plt.savefig(f"{auc_map}dim{dim}.png", bbox_inches='tight')
                plt.close()

def expectedK(k_values, counts):
    counts_np = np.array(counts)
    counts_np[counts_np <= 5] = 0
    densities_normed = counts_np / np.sum(counts_np)
    contributions = k_values * densities_normed
    expected_value = np.sum(contributions)

    return expected_value

def getPlotDict(overview_count_plot):
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

def createTable(plot_dict, metric_name, scaling_mode, save_map):
    key_pairs = np.array(list(plot_dict.keys()))
    sorted_dims = sorted(np.unique(key_pairs[:, 1]))
    sorted_auc = sorted(np.unique(key_pairs[:, 0]))
    auc_labels = [f"auc_{auc}\t" for auc in sorted_auc]
    table_array = np.zeros((len(sorted_auc), len(sorted_dims)))
    cell_colors = np.zeros((len(sorted_auc), len(sorted_dims)))
    for auc_index in sorted_auc:
        for dimension_index, dimension in enumerate(sorted_dims):
            expected_value = np.round(plot_dict[(auc_index, dimension)], 2)
            table_array[auc_index][dimension_index] = expected_value
            color = "red" if expected_value > 100 else "white"
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
def distanceBoxplot(problem_dict, save_map):
    for (metric_name, scaling), distance_dict in problem_dict.items():
        for auc_index, auc_dict in distance_dict.items():
            sub_map = f"{save_map}{metric_name}_{scaling}/auc{auc_index}/"
            Path(sub_map).mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(16, 8))
            all_dims = sorted(list(auc_dict.keys()))
            unique_dims = np.unique(all_dims).shape[0]
            #adjust = np.linspace(0.2, .8, unique_dims)
            for index, dim in enumerate(all_dims):
                distances = auc_dict[dim]
                grouped_distances = distances[:990, :].reshape(99, -1)
                plt.boxplot(distances.T)

                plt.xlabel("K-values")
                #plt.xscale("log")
                plt.ylabel("Distance")
                plt.savefig(f"{sub_map}dim{dim}.png", bbox_inches='tight')
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
                    apply(lambda x: x.nsmallest(n=5, columns='distance')).reset_index(drop=True)
                counted_data = filter_data.groupby("k_val").size()
                for k_val, counts in counted_data.items():
                    if k_val not in score_dict[auc_index][dim]:
                        score_dict[auc_index][dim][k_val] = counts
                    else:
                        score_dict[auc_index][dim][k_val] += counts
    return score_dict

metrics = ["pr", "dc"]
scalings = ["real_scaled", "fake_scaled"]
df_path = "./gaussian_dimension/dataframe.pkl"
df = pd.read_pickle(df_path)
auc_filter = [(0, 0.1), (0.1, 0.99), (0.99, 1)]
auc_filter = [(0, 0.1), (0.1, 0.9), (0.9, 1)]
print(df.info())
plotting=True
base_map = "./gaussian_dimension/paper_img/boxplot_auc_dimension/"
overview_map = "./gaussian_dimension/paper_img/overview_dimension/"
overview_dict = {}
if plotting:
    for metric_name in metrics:
        for scaling_mode in scalings:
            sel_data = df.loc[(df["metric_name"] == metric_name) & (df["scaling_mode"] == scaling_mode), :]
            sel_data = sel_data.loc[(sel_data["dimension"] <= 500), :]

            sub_map = f"{base_map}{metric_name}_{scaling_mode}/"
            count_dict = getOverview(sel_data, auc_filter)
            overview_dict[(metric_name, scaling_mode)] = count_dict
            #dfDistancePlots(sel_data, auc_filter, sub_map)

            plot_dict = getPlotDict(overview_dict)
            createTable(plot_dict, metric_name, scaling_mode, overview_map)



    #distanceBoxplot(problem_dict, base_map)
    #median_dict = getMedians(problem_dict)
    #plotCorrTable(median_dict, base_map)



