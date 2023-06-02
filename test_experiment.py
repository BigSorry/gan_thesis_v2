import numpy as np
import helper_functions as util
import check_densities as ch_den
import glob
import matplotlib.pyplot as plt
import matplotlib.transforms

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

def distancePlot(auc_data, save_path):
    plt.figure()
    for group_nr, mean_distances in auc_data.items():
        x = np.arange(mean_distances.shape[0]) + 1
        y = mean_distances
        plt.plot(x, y, label=f"auc_{group_nr}")
    plt.xscale("log")
    plt.xlabel("K-values")
    plt.ylabel("Distances")
    plt.legend()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
def corrPlot(auc_data, save_path):
    plt.figure()
    for group_nr, auc_plot_data in auc_data.items():
        x = np.array(auc_plot_data["x"])
        y = np.array(auc_plot_data["y"])
        index_sorted = np.argsort(x)
        x_sorted = x[index_sorted]
        y_sorted = y[index_sorted]
        plt.plot(x_sorted, y_sorted, label=f"auc_{group_nr}")
    plt.xscale("log")
    plt.xlabel("Dimension")
    plt.ylabel("Correlation distance vs K")
    plt.legend()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

path = f"./factors/pr/real_scaled/*.pkl"
metrics = ["pr", "dc"]
scalings = ["real", "fake"]
table_data = {}
for metric in metrics:
    for scaling in scalings:
        for file in glob.glob(path):
            dict = util.readPickle(file)
            auc_scores = dict["auc_scores"]
            distances = dict["distances"]
            sample_size = dict["experiment_config"]["samples"]
            dimension = dict["experiment_config"]["dimension"]
            iters = dict["experiment_config"]["iters"]
            if iters > 1:
                k_vals = [i for i in range(1, sample_size, 1)]
                k_scoring = np.zeros(len(k_vals))
                auc_filter = [(0, np.percentile(auc_scores, 25)), (np.percentile(auc_scores, 25), np.percentile(auc_scores, 75)),
                               (np.percentile(auc_scores, 75), 1), (0, 1)]
                for auc_index, (begin_auc, end_auc) in enumerate(auc_filter):
                    filter_distances = filterDistances(distances, auc_scores, begin_auc, end_auc)
                    sorted_indices = bestK(filter_distances)
                    updateDict(table_data, metric, scaling, dimension, auc_index, filter_distances)


map_images = "./gaussian_dimension/paper_img/plots/"
for (metric_name, scaling), dict_info in table_data.items():
    save_map = f"{map_images}/{metric_name}_{scaling}/"
    column_labels = [f"Top k{i}" for i in range(10)]
    auc_data = {i: {"x": [], "y": []} for i in range(4)}
    auc_data_distances = {i: [] for i in range(4)}
    for (dimension, auc_index), distances in dict_info.items():
        mean_vec = distances.mean(axis=1)
        correlation_coeff = np.corrcoef(mean_vec, k_vals)
        auc_data[auc_index]["x"].append(dimension)
        auc_data[auc_index]["y"].append(correlation_coeff[0, 1])
        auc_data_distances[auc_index] = mean_vec

    key = (metric_name, scaling)
    save_path = f"{map_images}{key}_distances.png"
    distancePlot(auc_data_distances, save_path)
    save_path = f"{map_images}{key}_corr.png"
    corrPlot(auc_data, save_path)




plt.show()