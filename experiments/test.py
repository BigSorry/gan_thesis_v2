import numpy as np
import matplotlib.pyplot as plt
from experiments import create_experiment as exp
from scipy.spatial import distance

def resample(needed_samples, outlier_distance, dimension):
    new_samples = []
    while len(new_samples) < needed_samples:
        mean_vec = np.zeros(dimension)
        identity_cov = np.eye(dimension)
        real_features = np.random.multivariate_normal(mean_vec, identity_cov, 5000)
        for vector in real_features:
            mal_distance = distance.mahalanobis(vector, mean_vec, identity_cov)
            if mal_distance > outlier_distance:
                new_samples.append(vector)
    new_samples = np.array(new_samples)
    return new_samples[:needed_samples, :]

def getData():
    sample_sizes = [5000]
    dimensions = [2, 8, 16, 32, 64, 512, 1024]
    for samples in sample_sizes:
        k_vals = np.array([1, 3, 7, 9, 16, 32])
        for index, dimension in enumerate(dimensions):
            mean_vec = np.zeros(dimension)
            identity_cov = np.eye(dimension)
            real_features = np.random.multivariate_normal(mean_vec, identity_cov, samples)
            distances = []
            for vector in real_features:
                mal_distance = distance.mahalanobis(vector, mean_vec, identity_cov)
                distances.append(mal_distance)
            distances = np.array(distances)
            outlier_value = np.quantile(distances, 0.95)
            print(outlier_value)
            boolean_mask = distances > outlier_value
            inliers = real_features[~boolean_mask, :]
            outliers = resample(inliers.shape[0], outlier_value, dimension)


            map_path = f"../gaussian_outlier/standard"
            pr_aboves, dc_aboves, pr_nearest_distances, dc_nearest_distances = exp.doExperiment("gaussian",
                                                                                                inliers,
                                                                                                outliers,
                                                                                                1, 1,
                                                                                                k_vals,
                                                                                                save_curve=True,
                                                                                                map_path=map_path)
            map_path = f"../gaussian_outlier/reverse/"
            pr_aboves, dc_aboves, pr_nearest_distances, dc_nearest_distances = exp.doExperiment("gaussian",
                                                                                                outliers,
                                                                                                inliers,
                                                                                                1, 1,
                                                                                                k_vals,
                                                                                                save_curve=True,
                                                                                                map_path=map_path)




getData()
plt.show()