import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utility_scripts import helper_functions as util
from utility_scripts import winner_pick_utility as win_util
from pathlib import Path

def excludeExperiment(metrics, scalings, filter_percentages, best_mode=True):
    all_result_dict = {}
    for scaling_mode in scalings:
        df_path_combined = f"../dataframe_evaluation/{scaling_mode}/combined/dataframe_all.pkl"
        all_dfs = util.readPickle(df_path_combined)
        for metric_name in metrics:
            scenario_key = (metric_name, scaling_mode)
            all_result_dict[scenario_key] = {}
            metric_df = all_dfs.loc[all_dfs["metric_name"] == metric_name, :]
            sel_data = metric_df.loc[(metric_df["dimension"] > 1) & (metric_df["dimension"] <= 555), :]
            for filter_percentage in filter_percentages:
                dataframe_top_picks = win_util.countTopPicks(sel_data, filter_percentage, best_mode)
                # Filtering selection
                excluded_data = dataframe_top_picks.loc[dataframe_top_picks["k_value"] == -1]
                experiment_keys_excluded = list(excluded_data.groupby(["iter", "dimension", "auc_score"]).groups.keys())
                all_result_dict[scenario_key][filter_percentage] = experiment_keys_excluded

    return all_result_dict

def saveKeys(result_dict, map_path):
    for (metric_name, scaling_mode), excluded_keys_dict in result_dict.items():
        save_path = f"{map_path}{metric_name}_{scaling_mode}.pkl"
        util.savePickle(save_path, excluded_keys_dict)


def smallExperiment():
    map_path = f"../filter_keys/"
    metrics = ["pr", "dc"]
    scalings = ["real_scaled", "fake_scaled"]
    # 0.002 is max 1 winner
    filter_percentages = [0.002, 0.01, 0.1, 0.25, 0.5, 0.75, 1.1]
    excluded_keys_dict = excludeExperiment(metrics, scalings, filter_percentages, best_mode=True)
    saveKeys(excluded_keys_dict, map_path)

smallExperiment()
