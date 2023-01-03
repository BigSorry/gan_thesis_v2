import numpy as np
import pandas as pd
import experiments_v2.helper_functions as util
import matplotlib.pyplot as plt
import metrics.likelihood_classifier as llc
import visualize as plotting
import experiments_v2.curve.llc_curve as llc

# knn and llc
def getDataframe(distribution_dict, k_vals):
    columns = ["key", "k_val",
               "precision", "recall", "density", "coverage"]
    row_data = []
    for base_param, base_data in distribution_dict.items():
        high_param = base_param[0]
        for other_key, other_data in distribution_dict.items():
            other_high_param = other_key[0]
            distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = util.getDistanceMatrices(base_data, other_data)
            for k_val in k_vals:
                boundaries_real = distance_matrix_real[:, k_val]
                boundaries_fake = distance_matrix_fake[:, k_val]
                precision, recall, density, coverage = util.getScores(distance_matrix_pairs, boundaries_fake,
                                                              boundaries_real, k_val)
                row = [(high_param, other_high_param), k_val,
                       precision, recall, density, coverage]
                row_data.append(row)

    dataframe = pd.DataFrame(columns=columns, data=row_data)
    return dataframe

def prepData():
    sample_size = 1000
    dimension = 2
    iters = 1
    k_vals = [1, 2, 4, 8, 16, 32, sample_size - 1]
    scale_factors = [0.1, 1]
    gaus_param = llc.getGausParams(scale_factors, dimension)
    param_dict = {"distribution_name": "gaus", "params": gaus_param}
    distribution_dict = llc.createDistributions(param_dict, sample_size, dimension, iters)
    dataframe = llc.getDataframe(distribution_dict, k_vals)

    return dataframe, distribution_dict

def doExperiment():
    dataframe, distribution_dict = prepData()
    x=0

doExperiment()