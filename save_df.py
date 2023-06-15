import numpy as np
import helper_functions as util
import glob
import pandas as pd

def filterDistances(distances, auc_scores, begin_auc, end_auc):
    boolean_vec = (np.array(auc_scores) >= begin_auc) & (np.array(auc_scores) <= end_auc)
    filter_distances = []
    for distance_row in distances:
        sel_values = np.array(distance_row)[boolean_vec]
        if len(sel_values) > 0:
            filter_distances.append(sel_values)

    return np.array(filter_distances)

def updateDict(dict, metric_name, scaling, dimension, auc_index, values_added):
    first_key = (metric_name, scaling)
    second_key = auc_index
    third_key = dimension
    if first_key not in dict:
        dict[first_key] = {}
    if second_key not in dict[first_key]:
        dict[first_key][second_key] = {}
    dict[first_key][second_key][third_key] = values_added

def addRows(pd_rows, metric_name, scaling_mode, dimension, auc_scores, distances):
    iter_array = (np.arange(auc_scores.shape[0])) // 50
    if iter_array.shape[0] > 500:
        print(iter_array.shape[0])
    for row_index, row_distances in enumerate(distances):
        for column_index, distance in enumerate(row_distances):
            auc_score = auc_scores[column_index]
            iter = iter_array[column_index]
            k_val = (row_index + 1)
            row_values = [metric_name, scaling_mode, iter, dimension, auc_score, k_val, distance]
            pd_rows.append(row_values)

# Assume 50 ratios taken and 10 iters
pd_rows = []
metrics = ["pr", "dc"]
scalings = ["real_scaled", "fake_scaled"]
column_names = ["metric_name", "scaling_mode", "iter", "dimension", "auc_score", "k_val", "distance"]
df_path = "./gaussian_dimension/dataframe.pkl"
for metric in metrics:
    for scaling in scalings:
        path = f"./factors/{metric}/{scaling}/*.pkl"
        for file in glob.glob(path):
            dict = util.readPickle(file)
            auc_scores = dict["auc_scores"]
            distances = dict["distances"]
            sample_size = dict["experiment_config"]["samples"]
            dimension = dict["experiment_config"]["dimension"]
            iters = dict["experiment_config"]["iters"]
            if dimension > 1 and dimension < 1000 and iters == 10:
                k_vals = [i for i in range(1, sample_size, 1)]
                k_scoring = np.zeros(len(k_vals))
                addRows(pd_rows, metric, scaling, dimension, auc_scores, distances)

dataframe = pd.DataFrame(data=pd_rows, columns=column_names)
print(dataframe.head())
dataframe.to_pickle(df_path)


