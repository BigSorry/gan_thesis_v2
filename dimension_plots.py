import numpy as np
import helper_functions as util
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.transforms
import pandas as pd
import seaborn as sns

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

def distanceBoxplotDF(dataframe, auc_filter, sub_map):
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
                    apply(lambda x: x.nsmallest(n=10, columns='distance')).reset_index(drop=True)
                grouped_data = filter_data.groupby("k_val")
                counted_data = grouped_data.size()
                x = list(grouped_data.groups.keys())
                k_vals = counted_data.index.values
                x = np.arange(k_vals.shape[0]) + 1
                y = grouped_data['distance'].apply(np.hstack).values
                y = counted_data.values
                #plt.boxplot(y.T, positions=x)
                plt.bar(x, y)
                plt.xticks(x, k_vals)
                plt.xlabel("K-values")
                if k_vals.shape[0] > 50:
                    plt.xscale("log")
                plt.ylabel("Count Top-10 per auc score")
                plt.savefig(f"{auc_map}dim{dim}.png", bbox_inches='tight')
                plt.close()

metrics = ["pr", "dc"]
metrics = ["pr"]
scalings = ["real_scaled", "fake_scaled"]
df_path = "./gaussian_dimension/dataframe.pkl"
dataframe = pd.read_pickle(df_path)
auc_filter = [(0, 0.1), (0.1, 0.9), (0.9, .99), (0.99, 1)]
print(dataframe.info())
plotting=True
base_map = "./gaussian_dimension/paper_img/boxplot_dimensions/"
if plotting:
    for metric_name in metrics:
        for scaling_mode in scalings:
            sel_data = dataframe.loc[(dataframe["metric_name"] == metric_name) & (dataframe["scaling_mode"] == scaling_mode), :]
            filter_dim = sel_data.loc[sel_data["dimension"] == 64, :]
            counter = filter_dim.groupby("auc_score").size()
            sub_map = f"{base_map}{metric_name}_{scaling_mode}/"
            distanceBoxplotDF(filter_dim, auc_filter, sub_map)



    #distanceBoxplot(problem_dict, base_map)
    #median_dict = getMedians(problem_dict)
    #plotCorrTable(median_dict, base_map)



