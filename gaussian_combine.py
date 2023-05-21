import numpy as np
import experiments.experiment_visualization as exp_vis

def combineResults(calc_dict):
    pr_distances = []
    dc_distances = []
    for factors, info_dict in calc_dict.items():
        pr_nearest_distances = info_dict["pr_distances"]
        dc_nearest_distances = info_dict["dc_distances"]
        pr_distances.append(pr_nearest_distances)
        dc_distances.append(dc_nearest_distances)
    pr_distances = np.array(pr_distances)
    dc_distances = np.array(dc_distances)

    pr_row = np.round([pr_distances.min(), pr_distances.mean(), pr_distances.max()], 2)
    dc_row = np.round([dc_distances.min(), dc_distances.mean(), dc_distances.max()], 2)

    return pr_row, dc_row
