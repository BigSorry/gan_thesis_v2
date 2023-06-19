import numpy as np
import helper_functions as util
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.transforms
import pandas as pd
import seaborn as sns

def expectedK(k_values, counts):
    counts_np = np.array(counts)
    counts_np[counts_np <= 5] = 0
    densities_normed = counts_np / np.sum(counts_np)
    contributions = k_values * densities_normed
    expected_value = np.sum(contributions)

    return expected_value

def overviewPlot(overview_count_plot, auc_filters, save_map):
    filters = len(auc_filters)
    fig, axs = plt.subplots(filters, 1)
    axs = axs.flatten()
    Path(save_map).mkdir(parents=True, exist_ok=True)
    plot_info = {}
    for (metric_name, scaling_mode), auc_dict in overview_count_plot.items():
        for auc_index, count_dict in auc_dict.items():
            if auc_index not in plot_info:
                plot_info[auc_index] = {}
            k_vals = list(count_dict.keys())
            counts = list(count_dict.values())
            expected_value_top_k = expectedK(k_vals, counts)
            plot_info[auc_index][(metric_name, scaling_mode)] = expected_value_top_k

    for auc_index, dict_results in plot_info.items():
        plt.figure()
        metric_scaling_pairs = list(dict_results.keys())
        yticks_str = [f"{pair[0]}_{pair[1]}" for pair in metric_scaling_pairs]
        y_vals = np.arange(len(yticks_str))
        expected_top_k = np.round(list(dict_results.values()), 1)
        plt.barh(y_vals, expected_top_k)
        plt.yticks(y_vals, yticks_str)
        plt.savefig(f"{save_map}overview_auc{auc_index}.png", bbox_inches='tight')
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
            dimension_data = sel_dataframe.loc[sel_dataframe["dimension"] == dim, :]
            if dimension_data.shape[0] > 0:
                filter_data = dimension_data.groupby(["iter", "auc_score"]).\
                    apply(lambda x: x.nsmallest(n=5, columns='distance')).reset_index(drop=True)
                counted_data = filter_data.groupby("k_val").size()
                for k_val, counts in counted_data.items():
                    if k_val not in score_dict[auc_index]:
                        score_dict[auc_index][k_val] = counts
                    else:
                        score_dict[auc_index][k_val] += counts
    return score_dict
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

metrics = ["pr", "dc"]
scalings = ["real_scaled", "fake_scaled"]
df_path = "./gaussian_dimension/dataframe.pkl"
df = pd.read_pickle(df_path)
auc_filter = [(0, 0.1), (0.1, 0.99), (0.99, 1)]
auc_filter = [(0, 0.1), (0.1, 0.9), (0.9, 1)]
print(df.info())
plotting=True
base_map = "./gaussian_dimension/paper_img/boxplot_auc/"
overview_map = "./gaussian_dimension/paper_img/overview/"
overview_dict = {}
if plotting:
    for metric_name in metrics:
        for scaling_mode in scalings:
            sel_data = df.loc[(df["metric_name"] == metric_name) & (df["scaling_mode"] == scaling_mode), :]
            #sel_data = sel_data.loc[(sel_data["dimension"] == 64), :]
            counter = sel_data.groupby("auc_score").size()
            sub_map = f"{base_map}{metric_name}_{scaling_mode}/"
            count_dict = getOverview(sel_data, auc_filter)
            overview_dict[(metric_name, scaling_mode)] = count_dict
            dfDistancePlots(sel_data, auc_filter, sub_map)

    overviewPlot(overview_dict, auc_filter, overview_map)




    #distanceBoxplot(problem_dict, base_map)
    #median_dict = getMedians(problem_dict)
    #plotCorrTable(median_dict, base_map)



