import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utility_scripts import helper_functions as util
from create_data_scripts import create_eval_df as eval_df
from utility_scripts.calc_util import getEvaluationPair
from pathlib import Path
import experiments.experiment_visualization as exp_vis

def plotByKey(metric_name, result_dict, keys, k_vals, map_path):
    for key in keys:
        lambda_factors = key[2]
        curve = result_dict[key]["curve"]
        eval_pairs = result_dict[key][metric_name+"_pairs"]
        label_name = metric_name + "_knn"

        plt.figure()
        title_text = f"Iter {key[0]} dimension {key[1]}\n lambda {key[2]:.4f} and auc{key[3]:.4f}"
        plt.title(title_text)
        exp_vis.plotTheoreticalCurve(curve, curve, lambda_factors, save=False)
        exp_vis.plotKNNMetrics(eval_pairs,  k_vals, label_name, "black", map_path, save=False)
        # plt.legend(loc='upper center', bbox_to_anchor=(0, 1),
        #            fancybox=True, shadow=True, ncol=1, fontsize=9)
        # Put a legend below current axis
        plt.legend(loc='upper center', bbox_to_anchor=(0.2, 1),
                  fancybox=True, shadow=True, ncol=1)
        Path(map_path).mkdir(parents=True, exist_ok=True)
        file_name = f"{metric_name}_{key}.png"
        plt.savefig(map_path+file_name, bbox_inches='tight')
        plt.close()

def filterDistance(result_dict, metric_name):
    sel_keys = []
    for experiment_key, experiment_dict in result_dict.items():
        column_name = metric_name + "_nearest_distances"
        distance_min_max = experiment_dict[column_name].max() - experiment_dict[column_name].min()
        if distance_min_max < 0.1:
            sel_keys.append(experiment_key)

    return sel_keys

def exampleFilter():
    iters = 1
    sample_size = 1000
    k_vals = [i for i in range(1, sample_size, 1)]
    dimension = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    #dimension = [512]
    df_path = "../dataframe_factors/dataframe_real.pkl"
    factors_saved = pd.read_pickle(df_path)
    dataframe_filtered = eval_df.filterByAUC(factors_saved)
    for dimension in dimension:
        dataframe_factors_sel = dataframe_filtered.loc[(dataframe_filtered["iter"] == 0) &
                                                  (dataframe_filtered["dimension"] == dimension), :]
        ratios = dataframe_factors_sel["ratio"].unique()
        #ratios = ratios[ratios < 0.1]
        dimension_transformed = dataframe_factors_sel["dimensions_transformed"].iloc[0]
        result_dict = getEvaluationPair(iters, k_vals, sample_size, dimension,
                                dimension_transformed, ratios, real_scaling=True, return_curve=True)


        pr_keys = filterDistance(result_dict, "pr")
        dc_keys = filterDistance(result_dict, "dc")

        plotByKey("pr", result_dict, pr_keys, k_vals, "./example_filter_extreme/pr/")
        plotByKey("dc", result_dict, dc_keys, k_vals, "./example_filter_extreme/dc/")

exampleFilter()
