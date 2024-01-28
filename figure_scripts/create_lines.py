import numpy as np
from utility_scripts import helper_functions as util
from utility_scripts import winner_pick_utility as win_util
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from matplotlib import cm
import matplotlib

def addTable(table_info, variable_name):
    xmin, ymin, width, height = 0.2, -0.45, .9, 0.1*3
    table_vals = np.array(table_info).T
    colors = np.full(table_vals.shape, "000000")
    # colors[0, :] = "000000"
    # colors[2, table_vals[2, :] >= 10] = "000000"
    table = plt.table(cellText=table_vals,
                      rowLabels=[variable_name, "Lowest distance", "Corresponding k-value"],
                      #colLabels=[variable_name, "Lowest distance", "Corresponding k-value"],
                      cellLoc="center",
                      #cellColours=colors,
                      bbox=(xmin, ymin, width, height))
    table.set_fontsize(12)

def lowestPoints(distances):
    offset = 1e-3
    top_distance = distances.min()
    lowest_points = distances[distances <= (top_distance + offset)]
    for index, value in lowest_points.iteritems():
        k_value = index
        distance = value
        plt.scatter(k_value, distance, color='red')
        plt.annotate(f'({k_value:.2f}, {distance:.2f})', xy=(k_value, distance), xytext=(k_value, distance + 0.5),
                     arrowprops=dict(facecolor='black', shrink=0.05))

def plotLines(variable_name, df_grouped, quantile_factor, map_path, extra_name="", include_table=False):
    # normalize item number values to colormap
    total_groups = df_grouped.ngroups
    #cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["yellow", "red"])
    cmap = cm.get_cmap('YlOrRd')
    colors = cmap(np.linspace(0.2, 1, total_groups))
    if variable_name == "Dimension":
        colors = cmap(np.linspace(0.01, 1, total_groups))
        # dimension_values = np.array(list(df_grouped.groups.keys()))
        # my_transformed_range = (dimension_values - np.min(dimension_values))\
        #                        / (np.max(dimension_values) - np.min(dimension_values))
        # colors = [cmap(min(value+0.3, 1)) for value in my_transformed_range]

    quantile = int(quantile_factor*100)
    plt.title(f"Quantile {quantile}th distance {extra_name}")
    for index, (variable_value, grouped_df) in enumerate(df_grouped):
        quantile_distance = grouped_df.groupby("k_val")["distance"].quantile(q=quantile_factor)
        x_vals = np.arange(quantile_distance.shape[0]) + 1
        dimension_color = colors[index]
        plt.plot(x_vals, quantile_distance, c=dimension_color, label=f"{variable_name} {variable_value}")
        lowestPoints(quantile_distance)

    plt.legend(fontsize="x-small", ncol=2)
    plt.ylim([0, 1.3])
    plt.xscale("log")
    Path(map_path).mkdir(parents=True, exist_ok=True)
    save__path = f"{map_path}q{quantile_factor}{extra_name}.png"
    plt.savefig(save__path, dpi=300, bbox_inches="tight")
    plt.close()

def plotNoGroupLines(dataframe, quantile_factor):
    #quantile = int(quantile_factor * 100)
    #plt.title(f"Quantile {quantile}th distance")
    quantile_distance = dataframe.groupby("k_val")["distance"].quantile(q=quantile_factor)
    x_vals = np.arange(quantile_distance.shape[0]) + 1
    plt.plot(x_vals, quantile_distance, c="black", label=f"All AUC included")

def createPlots(metrics, scalings, max_winner, quantiles, max_dim, parent_map):
    # save sub map if needed
    auc_bins = [0, 0.1, 0.45, 0.9, 1.1]
    for scaling_mode in scalings:
        df_path_combined = f"../dataframe_evaluation/{scaling_mode}/combined/dataframe_all.pkl"
        all_dfs = util.readPickle(df_path_combined)
        for metric_name in metrics:
            scenario_str = f"{metric_name}_{scaling_mode}_max_winner{max_winner}"
            metric_df = all_dfs.loc[all_dfs["metric_name"] == metric_name, :]
            sel_data = metric_df.loc[(metric_df["dimension"] > 1) & (metric_df["dimension"] <= max_dim), :]
            filter_mask = win_util.getFilterMask(sel_data, max_winner)
            filtered_data = sel_data.iloc[filter_mask, :].copy()
            filtered_data.loc[:, "auc_groups"] = pd.cut(filtered_data["auc_score"], auc_bins, include_lowest=True, right=False)
            dimension_grouped = filtered_data.groupby(["dimension"])
            auc_grouped = filtered_data.groupby(["auc_groups"])
            #auc_dim_grouped = filtered_data.groupby(["dimension", "auc_groups"])


            for quant in quantiles:
                map_path = f"{parent_map}/dimension/{scenario_str}/"
                #plt.figure(figsize=(20, 5))
                plotLines("Dimension", dimension_grouped, quant, map_path, include_table=False)
                map_path = f"{parent_map}/auc/{scenario_str}/"
                plt.figure()
                plotNoGroupLines(filtered_data, quant)
                plotLines("AUC", auc_grouped, quant, map_path)

                # for dim, grouped_df in dimension_grouped:
                #     auc_dim_grouped = grouped_df.groupby(["auc_groups"])
                #     map_path = f"{parent_map}/auc_dimension/{scenario_str}_q{quant}/"
                #     plotNoGroupLines(grouped_df, quant)
                #     plotLines("auc_dimension", auc_dim_grouped, quant, map_path, extra_name=f"_dim_{dim}", include_table=True)

def allExperiments():
    metrics = ["pr", "dc"]
    scalings = ["real_scaled", "fake_scaled"]
    # Max 1 winner
    max_winners = [1]
    quantiles = [0.25, 0.5, 0.75]
    max_dim = 555
    for max_winner in max_winners:
        parent_map = f"../experiment_figures/lines/"
        createPlots(metrics, scalings, max_winner, quantiles, max_dim, parent_map)

def smallExperiment():
    metrics = ["pr", "dc"]
    scalings = ["real_scaled"]
    max_winner = 1
    max_dim = 555
    quantiles = [0.25, 0.5, 0.75]
    parent_map = f"../experiment_figures/lines/"
    createPlots(metrics, scalings, max_winner, quantiles, max_dim, parent_map)

allExperiments()
