import numpy as np
import helper_functions as util
import check_densities as ch_den
import glob

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

path = f"./factors/pr/real_scaled/*.pkl"
box_map = "./gaussian_dimension/paper_img/boxplots/"
metrics = ["pr", "dc"]
scalings = ["real", "fake"]
for metric in metrics:
    for scaling in scalings:
        save_map = f"{box_map}/{metric}_{scaling}/"
        for file in glob.glob(path):
            dict = util.readPickle(file)
            auc_scores = dict["auc_scores"]
            distances = dict["distances"]
            sample_size = dict["experiment_config"]["samples"]
            dimension = dict["experiment_config"]["dimension"]
            iters = dict["experiment_config"]["iters"]
            if iters > 5 and dimension > 200:
                k_vals = [i for i in range(1, sample_size, 1)]
                k_scoring = np.zeros(len(k_vals))
                auc_filter = [(0, np.percentile(auc_scores, 25)), (np.percentile(auc_scores, 25), np.percentile(auc_scores, 75)),
                               (np.percentile(auc_scores, 75), 1), (0, 1)]
                for (begin_auc, end_auc) in auc_filter:
                    filter_distances = filterDistances(distances, auc_scores, begin_auc, end_auc)
                    sorted_indices = bestK(filter_distances)
                    k_scoring += sorted_indices

                    #k_scoring_sorted = np.argsort(k_scoring)
                    print(sorted_indices[:10])
                    print()


