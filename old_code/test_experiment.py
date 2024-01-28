import numpy as np
from utility_scripts import helper_functions as util
import glob
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

def plotTable(distances, save_path):
    distances = np.array(distances)
    mean_vecs = distances.mean(axis=1)
    plt.figure(figsize=(14, 6))
    #plt.boxplot(distances.T, positions=k_vals)
    plt.plot(k_vals, mean_vecs)
    plt.ylim([0, 1.1])
    plt.xlabel("K-value")
    plt.xscale("log")
    plt.xticks(rotation=90)
    # plt.xticks(np.arange(select_indices.shape[0])+1, taken_k)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def filterDistances(distances, auc_scores, begin_auc, end_auc):
    boolean_vec = (np.array(auc_scores) >= begin_auc) & (np.array(auc_scores) <= end_auc)
    filter_distances = []
    for distance_row in distances:
        sel_values = np.array(distance_row)[boolean_vec]
        if len(sel_values) > 0:
            filter_distances.append(sel_values)

    return np.array(filter_distances)

def bestK(distances):
    smallest_distances = np.mean(distances, axis=1)
    sorted_indices = np.argsort(smallest_distances)

    return sorted_indices

def updateDict(dict, metric_name, scaling, dimension, auc_index, values_added):
    first_key = (metric_name, scaling)
    second_key = (dimension, auc_index)
    if first_key not in dict:
        dict[first_key] = {}
    if second_key not in dict[first_key]:
        dict[first_key][second_key] = []
    dict[first_key][second_key] = values_added

def bestKPlot(best_k_dict, save_map):
    for group_nr, best_indices in best_k_dict.items():
        all_dims = sorted(list(best_indices.keys()))
        plt.figure()
        for dim in all_dims:
            best_ks = best_indices[dim]
            unique, counts = np.unique(best_ks, return_counts=True)
            plt.bar(unique, counts)
        plt.xlabel("K-values")
        plt.xscale("log")
        plt.ylabel("Count Best value")
        plt.savefig(f"{save_map}auc_nr_{group_nr}.png", bbox_inches='tight')
        plt.close()

def distanceBoxplot(auc_data, save_map):
    for group_nr, distance_dict in auc_data.items():
        plt.figure()
        list_distances = list(distance_dict.values())
        distances_np = np.hstack(list_distances)
        plt.boxplot(distances_np.T)
        plt.xscale("log")
        plt.xlabel("K-values")
        plt.ylabel("Distances")
        plt.savefig(f"{save_map}auc_nr{group_nr}.png", bbox_inches='tight')
        plt.close()
def distancePlot(auc_data, save_map):
    for group_nr, distance_dict in auc_data.items():
        plt.figure()
        all_dims = sorted(list(distance_dict.keys()))
        for dim in all_dims:
            mean_distances = distance_dict[dim]
            x = np.arange(mean_distances.shape[0]) + 1
            y = mean_distances
            plt.plot(x, y, label=f"dim_{dim}")
        plt.xscale("log")
        plt.xlabel("K-values")
        plt.ylabel("Distances")
        plt.ylim([0, 1])
        plt.legend()
        plt.savefig(f"{save_map}auc_nr{group_nr}.png", bbox_inches='tight')
        plt.close()
def corrPlot(auc_data, save_map):
    for group_nr, auc_plot_data in auc_data.items():
        plt.figure()
        x = np.array(auc_plot_data["x"])
        y = np.array(auc_plot_data["y"])
        index_sorted = np.argsort(x)
        x_vals = (np.arange(x.shape[0]) + 1)
        x_label = x[index_sorted]
        y_sorted = y[index_sorted]
        plt.bar(x_vals, y_sorted, label=f"auc_{group_nr}")
        plt.xticks(x_vals, x_label)
        plt.xlabel("Dimension")
        plt.ylabel("Correlation distance vs K")
        plt.ylim([0, 1])
        plt.legend()
        plt.savefig(f"{save_map}auc_nr{group_nr}_corr.png", bbox_inches='tight')
        plt.close()
def plotAucScores(auc_scores):
    plt.figure()
    sns.histplot(auc_scores, bins=10, stat="probability")
    plt.ylabel("Probability")
    plt.xlabel("AUC score")

metrics = ["pr", "dc"]
scalings = ["real_scaled", "fake_scaled"]
table_data = {}
sel_dimension = [2, 64, 512]
all_auc = []
auc_filter = [(0, 0.1), (0.1, 0.9), (0.9, 1), (0, 1)]
for metric in metrics:
    for scaling in scalings:
        if scaling == "real_scaled":
            path_factors = "../dataframe_factors/dataframe_real.pkl"
        else:
            path_factors = "../dataframe_factors/dataframe_fake.pkl"
        for file in glob.glob(path_factors):
            dict = util.readPickle(file)
            auc_scores = dict["auc_scores"]
            all_auc.extend(auc_scores)
            distances = dict["distances"]
            sample_size = dict["experiment_config"]["samples"]
            dimension = dict["experiment_config"]["dimension"]
            iters = dict["experiment_config"]["iters"]
            if dimension > 1  and iters == 10:
                k_vals = [i for i in range(1, sample_size, 1)]
                k_scoring = np.zeros(len(k_vals))
                for auc_index, (begin_auc, end_auc) in enumerate(auc_filter):
                    filter_distances = filterDistances(distances, auc_scores, begin_auc, end_auc)
                    updateDict(table_data, metric, scaling, dimension, auc_index, filter_distances)


plotting=True
if plotting:
    base_map_distance = "./test/distance_plots/"
    base_map_distance_box = "./test/boxplots/"
    base_map_best_k = "./test/best_k/"
    for (metric_name, scaling), dict_info in table_data.items():
        column_labels = [f"Top k{i}" for i in range(10)]
        auc_data = {i: {"x": [], "y": []} for i in range(4)}
        auc_data_distances = {i: {} for i in range(4)}
        all_data_distances = {i: {} for i in range(4)}
        best_k_runs = {i: {} for i in range(4)}

        # Distance dimension -> k_val x experiment run
        for (dimension, auc_index), distances in dict_info.items():
            if distances.shape[0] > 0:
                mean_vec = distances.mean(axis=1)
                best_k = distances.argmin(axis=0) + 1
                correlation_coeff = np.corrcoef(mean_vec, k_vals)
                # auc_data[auc_index]["x"].append(dimension)
                # auc_data[auc_index]["y"].append(correlation_coeff[0, 1])
                auc_data_distances[auc_index][dimension] = mean_vec
                all_data_distances[auc_index][dimension] = distances
                best_k_runs[auc_index][dimension] = best_k


        key_str = f"{metric_name}_{scaling}"
        sub_map_distance = f"{base_map_distance}{key_str}/"
        sub_map_distance_boxplot = f"{base_map_distance_box}{key_str}/"
        sub_map_best_k = f"{base_map_best_k}{key_str}/"
        Path(sub_map_distance).mkdir(parents=True, exist_ok=True)
        Path(sub_map_distance_boxplot).mkdir(parents=True, exist_ok=True)
        Path(sub_map_best_k).mkdir(parents=True, exist_ok=True)
        distanceBoxplot(all_data_distances, sub_map_distance_boxplot)
        distancePlot(auc_data_distances, sub_map_distance)
        bestKPlot(best_k_runs, sub_map_best_k)
        #corrPlot(auc_data, sub_map)


plotAucScores(all_auc)
plt.show()

