import helper_functions as helper
import numpy as np
import helper_functions as util
def saveDistributions(param_dict, save_path):
    iterations = param_dict['iterations']
    sample_sizes = param_dict['sample_sizes']
    dimensions = param_dict['dimensions']
    lambda_factors = param_dict['lambda_factors']
    distribution_dict = {}
    for iter in range(iterations):
        for sample_size in sample_sizes:
            for dimension in dimensions:
                for scale in lambda_factors:
                    key = (iter, sample_size, dimension, scale)
                    mean_vec = np.zeros(dimension)
                    identity_cov = np.eye(dimension)*scale
                    samples = np.random.multivariate_normal(mean_vec, identity_cov, sample_size)
                    distribution_dict[key] = samples

    helper.savePickle(save_path, distribution_dict)
def saveDistances(distribution_dict, save_path):
    distance_matrix_dict = {}
    for key, samples in distribution_dict.items():
        (iter, sample_size, dimension, scale) = key
        if scale == 1:
            distance_matrix_dict[key] = {}
            for other_key , other_samples in distribution_dict.items():
                (other_iter, other_sample_size, other_dimension, Other_scale) = other_key
                if iter == other_iter and sample_size == other_sample_size and dimension == other_dimension:
                    distance_matrix_dict[key][other_key] = {}
                    distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = util.getDistanceMatrices(samples, other_samples)
                    distance_matrix_dict[key][other_key]["real"] = distance_matrix_real
                    distance_matrix_dict[key][other_key]["fake"] = distance_matrix_fake
                    distance_matrix_dict[key][other_key]["real_fake"] = distance_matrix_pairs
    helper.savePickle(save_path, distance_matrix_dict)
def saveData(save_path_distributions, save_path_distances, param_dict):
    saveDistributions(param_dict, save_path_distributions)
    distribution_dict = helper.readPickle(save_path_distributions)
    saveDistances(distribution_dict, save_path_distances)