import numpy as np
import matplotlib.pyplot as plt
from experiments import create_experiment as exp

def getData():
    sample_sizes = [1000]
    dimensions = [2]
    outlier_value = 0 + np.sqrt(1)
    for samples in sample_sizes:
        k_vals = np.array([1, 3, 7, 9, 16, 32, 64, 128])
        for dimension in dimensions:
            real_features = np.random.normal(loc=0.0, scale=1.0,
                                             size=[samples, dimension])

            fake_features = np.random.normal(loc=0.0, scale=1.0,
                                             size=[samples, dimension])

            outlier_boolean = (np.abs(real_features) > outlier_value).any(axis=1)
            real_inliers = real_features[~outlier_boolean, :]
            real_outlier = real_features[outlier_boolean, :]

            plt.figure()
            plt.scatter(real_outlier[:, 0], real_outlier[:, 1], c="red")
            plt.scatter(real_inliers[:, 0], real_inliers[:, 1], c="blue")
            plt.show()
            map_path = f"../gaussian_outlier/"
            pr_aboves, dc_aboves, pr_nearest_distances, dc_nearest_distances = exp.doExperiment("gaussian",
                                                                                                real_features,
                                                                                                real_inliers,
                                                                                                1, 1,
                                                                                                k_vals,
                                                                                                save_curve=True,
                                                                                                map_path=map_path)


getData()