import numpy as np
import helper_functions as util
import check_densities as ch_den
import glob
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

def updateDict(dict, metric_name, scaling, dimension, values_added):
    first_key = (metric_name, scaling)
    second_key = (dimension)
    if first_key not in dict:
        dict[first_key] = {}
    if second_key not in dict[first_key]:
        dict[first_key][second_key] = []
    dict[first_key][second_key] = values_added

def distancePlotLine(problem_dict, save_map):
    Path(save_map).mkdir(parents=True, exist_ok=True)
    all_means = []
    colors = ["blue", "black", "red", "green"]
    error_every_vals = [2, 2, 2, 2]
    index = 0
    plt.figure(figsize=(16,8))
    for (metric_name, scaling), distance_dict in problem_dict.items():
        list_distances = list(distance_dict.values())
        distances_np = np.hstack(list_distances)
        mean_vec = distances_np.mean(axis=1)
        std_vec = distances_np.std(axis=1)
        median = np.quantile(distances_np, q=0.5, axis=1)
        all_means.append(mean_vec)
        color_sel = colors[index]
        error_every = error_every_vals[index]
        index += 1
        x = np.arange(mean_vec.shape[0]) + 1
        # plt.plot(x, median, '-', c=color_sel)
        # plt.fill_between(x, median - std_vec, median + std_vec,
        #                  alpha=alpha_val, edgecolor=color_sel, facecolor=color_sel)
        plt.errorbar(x, median, std_vec,
                     label=f"{metric_name}_{scaling}", c=color_sel, elinewidth=1, marker="8", errorevery=error_every)

    plt.xlabel("K-values")
    plt.xscale("log")
    plt.ylabel("Median distance")
    plt.legend(loc="upper left")
    plt.savefig(f"{save_map}plot_medians.png", bbox_inches='tight')
    plt.close()


path = f"./factors/pr/real_scaled/*.pkl"
metrics = ["pr", "dc"]
scalings = ["real_scaled", "fake_scaled"]
problem_dict = {}
auc_filter = [(0, 0.3), (0.3, 0.7), (0.7, 1), (0, 1)]
distances_included = 10
for metric in metrics:
    for scaling in scalings:
        path = f"./factors/{metric}/{scaling}/*.pkl"
        for file in glob.glob(path):
            dict = util.readPickle(file)
            distances = dict["distances"]
            sel_distances = distances[:, :distances_included]
            sample_size = dict["experiment_config"]["samples"]
            dimension = dict["experiment_config"]["dimension"]
            iters = dict["experiment_config"]["iters"]
            if dimension > 1 and iters == 10:
                updateDict(problem_dict, metric, scaling, dimension, distances)


plotting=True
if plotting:
    base_map = "./gaussian_dimension/paper_img/plot_all/"
    distancePlotLine(problem_dict, base_map)




