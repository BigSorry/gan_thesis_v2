import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utility_scripts import helper_functions as util
from utility_scripts import winner_pick_utility as win_util
from pathlib import Path

def getDict(filter_dataframe_top_picks):
    grouped_data = filter_dataframe_top_picks.groupby(["iter", "dimension", "auc_score"])
    auc_scores = []
    dimensions = []
    distances = []
    k_values_winners = []
    for key_data, experiment_data in grouped_data:
        dimension = key_data[1]
        auc_score = key_data[2]
        distance = experiment_data["distance"].max()
        k_values = experiment_data["k_value"].values

        dimensions.append(dimension)
        auc_scores.append(auc_score)
        distances.append(distance)
        k_values_winners.extend(k_values)

    result_dict = {"k_value": k_values_winners, "distance": distances,
                                                "auc_score": auc_scores, "dimension": dimensions}
    return result_dict

def getExclusionPercentage(df_original, df_excluded):
    experiment_amount = df_original.groupby(["iter", "dimension", "auc_score"]).ngroups
    experiment_amount_excluded = df_excluded.groupby(["iter", "dimension", "auc_score"]).ngroups
    excluded_percentage = (experiment_amount_excluded / experiment_amount) * 100

    return np.round(excluded_percentage, 2)

def getResults(metrics, scalings, max_winners, excluded_analysis, best_mode=True):
    all_result_dict = {}
    for scaling_mode in scalings:
        df_path_combined = f"../dataframe_evaluation/{scaling_mode}/combined/dataframe_all.pkl"
        all_dfs = util.readPickle(df_path_combined)
        for metric_name in metrics:
            metric_df = all_dfs.loc[all_dfs["metric_name"] == metric_name, :]
            sel_data = metric_df.loc[(metric_df["dimension"] > 1) & (metric_df["dimension"] <= 555), :]
            for max_winner in max_winners:
                dataframe_top_picks = win_util.countTopPicks(sel_data, max_winner, best_mode)
                # Filtering selection
                filter_dataframe_top_picks_included = dataframe_top_picks.loc[dataframe_top_picks["k_value"] > -1]
                filter_dataframe_top_picks_excluded = dataframe_top_picks.loc[dataframe_top_picks["k_value"] == -1]

                result_dict = getDict(filter_dataframe_top_picks_included)
                excluded_percentage = getExclusionPercentage(dataframe_top_picks, filter_dataframe_top_picks_excluded)
                result_dict["excluded"] = excluded_percentage
                dict_key = (metric_name, scaling_mode)
                if dict_key not in all_result_dict:
                    all_result_dict[dict_key] = {}
                all_result_dict[dict_key][max_winner] = result_dict

    return all_result_dict

def textBar(bars, first_bar=True):
    for bar in bars:
        x_loc = bar.get_x() + bar.get_width()
        if first_bar:
            x_loc = bar.get_x()
        plt.annotate(f'{bar.get_height()}',
                     (x_loc, bar.get_height() + 0.02),
                     verticalalignment='bottom', horizontalalignment='center',
                     fontsize=9)

def saveColumnPlot(all_df, excluded_df, column_name, title_text, save_path):
    column_values = sorted(all_df[column_name].unique())
    all_experiments = []
    included_experiments = []
    for column_value in column_values:
        sel_old_data = all_df.loc[all_df[column_name] == column_value, :]
        sel_excluded_data = excluded_df.loc[excluded_df[column_name] == column_value, :]
        experiments_nr = sel_old_data.groupby(["iter", "dimension", "auc_score"]).ngroups
        excluded_data = sel_excluded_data.groupby(["iter", "dimension", "auc_score"]).ngroups
        experiments_included_nr = experiments_nr - excluded_data
        all_experiments.append(experiments_nr)
        included_experiments.append(experiments_included_nr)

    x_values = (np.arange(len(column_values)) + 1) * 2
    offset = 0.5
    width = 0.5
    plt.figure()
    plt.title(title_text)
    bars = plt.bar(x_values - offset, all_experiments, width, label="All experiment number")
    bars2 = plt.bar(x_values, included_experiments, width, label="Included experiment number")
    plt.xticks(x_values, column_values)
    textBar(bars, first_bar=True)
    textBar(bars2, first_bar=False)
    plt.legend()

    #plt.show()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def excludeExperiment(metrics, scalings, filter_percentages, map_path, best_mode=True):
    all_result_dict = {}
    auc_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
    for scaling_mode in scalings:
        df_path_combined = f"../dataframe_evaluation/{scaling_mode}/combined/dataframe_all.pkl"
        all_dfs = util.readPickle(df_path_combined)
        for metric_name in metrics:
            metric_df = all_dfs.loc[all_dfs["metric_name"] == metric_name, :]
            sel_data = metric_df.loc[(metric_df["dimension"] > 1) & (metric_df["dimension"] <= 4), :]
            for filter_percentage in filter_percentages:
                dataframe_top_picks = win_util.countTopPicks(sel_data, filter_percentage, best_mode)
                dataframe_top_picks["auc_groups"] = pd.cut(dataframe_top_picks["auc_score"], auc_bins,
                                                           include_lowest=True, right=False)
                # Filtering selection
                filter_dataframe_top_picks_excluded = dataframe_top_picks.loc[dataframe_top_picks["k_value"] == -1]
                excluded_percentage = getExclusionPercentage(dataframe_top_picks, filter_dataframe_top_picks_excluded)
                title_text = f"{metric_name, scaling_mode, filter_percentage}\n "\
                             f"Experiments excluded:\n"\
                             f" {excluded_percentage}%"
                save_path = f"{map_path}/dimension/{metric_name}_{scaling_mode}_{filter_percentage}.png"
                Path(map_path).mkdir(parents=True, exist_ok=True)
                saveColumnPlot(dataframe_top_picks, filter_dataframe_top_picks_excluded, "dimension",
                                  title_text, save_path)
                save_path = f"{map_path}/auc_score/{metric_name}_{scaling_mode}_{filter_percentage}.png"
                saveColumnPlot(dataframe_top_picks, filter_dataframe_top_picks_excluded, "auc_groups",
                                  title_text, save_path)

    return all_result_dict

def plotDict(all_result_dict, save_map_path, name_values):
    for (metric_name, scaling_mode), result_dict in all_result_dict.items():
        plt.figure(figsize=(22, 6))
        filter_options = len(result_dict)
        for index, (max_winner, values_dict) in enumerate(result_dict.items()):
            subplots = plt.subplot(1, filter_options, index + 1)
            boxplot_values = values_dict[name_values]
            excluded_percentage = values_dict["excluded"]
            plt.title(f"Max winner per experiment:\n"
                      f" {max_winner}\n "
                      f"Experiments excluded:\n"
                      f" {excluded_percentage:.2f}%")
            plt.ylabel("K_value")
            plt.boxplot(boxplot_values)
            if name_values == "k_value":
                subplots.set_yscale("log")
            elif name_values == "distance":
                subplots.set_ylim(bottom=0, top=1)
            elif name_values == "auc_score":
                subplots.set_ylim(bottom=0, top=1.1)
            elif name_values == "dimension":
                subplots.set_ylim(bottom=1, top=515)

        Path(save_map_path).mkdir(parents=True, exist_ok=True)
        save_path = f"{save_map_path}/{metric_name}_{scaling_mode}_{name_values}"
        plt.savefig(save_path +".pdf", format="pdf", bbox_inches='tight', dpi=150)
        plt.savefig(save_path+".png", bbox_inches="tight", dpi=150)
        plt.close()

def allExperiment(included_analysis=True):
    if included_analysis:
        map_path = f"./filter_analysis/included_analysis/"
    else:
        map_path = f"./filter_analysis/excluded_analysis/"
    metrics = ["pr", "dc"]
    scalings = ["real_scaled", "fake_scaled"]
    # 0.002 is max 1 winner
    filter_percentages = [0.002, 0.01, 0.1, 0.25, 0.5, 0.75, 1.1]
    max_winners = [1, 10, 100, 250, 500, 750, 999]
    if included_analysis:
        value_names = ["k_value", "distance"]
        result_dict = getResults(metrics, scalings, max_winners, included_analysis, best_mode=True)
        for value_name in value_names:
            save_map_path = f"{map_path}{value_name}/"
            plotDict(result_dict, save_map_path, value_name)
    else:
        excludeExperiment(metrics, scalings, max_winners, map_path, best_mode=True)

def smallExperiment(included_analysis=True):
    if included_analysis:
        map_path = f"./filter_analysis/included_analysis/"
    else:
        map_path = f"./filter_analysis/excluded_analysis/"
    metrics = ["pr"]
    scalings = ["real_scaled"]
    # 0.002 is max 1 winner
    max_winners = [1, 10, 100, 250, 500, 750, 999]

    if included_analysis:
        value_names = ["k_value", "distance"]
        result_dict = getResults(metrics, scalings, max_winners, included_analysis, best_mode=True)
        for value_name in value_names:
            save_map_path = f"{map_path}{value_name}/"
            plotDict(result_dict, save_map_path, value_name)

    else:
        excludeExperiment(metrics, scalings, max_winners, map_path, best_mode=True)

smallExperiment(included_analysis=True)
