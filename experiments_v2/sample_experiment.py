import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visualize as plotting
import experiments_v2.helper_functions as util

def getKParams(sample_size):
    vals = []
    for i in range(1, 10):
        size = int(sample_size * (i/10))
        vals.append(size)
    vals.append(sample_size-1)
    return vals

def doVolumeExperiment():
    # Setup
    sample_sizes = [10, 100, 1000, 2000]
    dimension = 2
    mean = np.zeros(dimension)
    scale_factors = [0.01, 0.1, 1, 10, 100, 1000, 10000, 10**6]
    recalls = {i:[] for i in scale_factors}
    coverages = {i:[] for i in scale_factors}
    columns = ["sample_size", "dimension", "lambda", "k_val", "recall", "coverage"]
    row_data = []
    for samples in sample_sizes:
        k_vals = getKParams(samples)
        print(k_vals)
        for scale_factor in scale_factors:
            cov_real = np.eye(dimension) * scale_factor
            cov_fake = np.eye(dimension)
            real_features = np.random.multivariate_normal(mean, cov_real, size=samples)
            fake_features = np.random.multivariate_normal(mean, cov_fake, size=samples)
            distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = util.getDistanceMatrices(real_features, fake_features)
            for k_val in k_vals:
                # Calculations
                boundaries_real = distance_matrix_real[:, k_val]
                boundaries_fake = distance_matrix_real[:, k_val]
                precision, recall, density, coverage = util.getScores(distance_matrix_pairs, boundaries_fake, boundaries_real, k_val)
                recalls[scale_factor].append(recall)
                coverages[scale_factor].append(coverage)
                row = [samples, dimension, scale_factor, k_val, recall, coverage]
                row_data.append(row)

    datafame = pd.DataFrame(columns=columns, data=row_data)
    for samples in sample_sizes:
        select_data = datafame.loc[datafame["sample_size"] == samples, :]
        x = np.round(select_data["k_val"] / samples, 2)
        recalls = select_data["recall"].values
        coverages = select_data["coverage"].values

        plt.figure()
        plt.title(samples)
        plt.bar(x, recalls,  label='Recall')
        plt.bar(x, coverages, label='Coverages')


