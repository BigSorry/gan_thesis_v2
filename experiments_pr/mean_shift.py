import numpy as np
import visualize as plotting
import experiment_utils as util
import metrics.likelihood_classifier as llc
import metrics.metrics as mtr

def doShitfs(sample_count, dimension, mean_shift,
                          method_name, params):
    mean_vec = np.zeros(dimension)
    cov_matrix = np.eye(dimension)
    real_features = np.random.multivariate_normal(mean_vec, cov_matrix, size=sample_count)
    shifted_mean = mean_vec + mean_shift
    fake_features = np.random.multivariate_normal(shifted_mean, cov_matrix, size=sample_count)
    pr_score, curve, cluster_labels = mtr.getPRCurve(real_features, fake_features, method_name, params)

    return pr_score, curve, cluster_labels

def addNoise(path, sample_count, dims, metric_params, stds, save=False):
    curve_method_names, lambdas = util.prepData()
    curve_dict = {name: {} for name in curve_method_names}
    for dim in dims:
        curve_dict[curve_method_names[0]][dim] = {}
        curve_dict[curve_method_names[1]][dim] = {}
        mean_vec = np.zeros(dim)
        cov_matrix = np.eye(dim)
        real_features = np.random.multivariate_normal(mean_vec, cov_matrix, size=sample_count)

        for val in stds:
            translation_vec = np.random.normal(0, val, size=mean_vec.shape[0])
            print(translation_vec)
            shifted_mean = mean_vec + translation_vec
            fake_features = np.random.multivariate_normal(shifted_mean, cov_matrix, size=sample_count)
            pr_score, curve_histo, cluster_labels = mtr.getHistoPR(real_features, fake_features, cluster_param=metric_params[0])
            # Likelihood classifier
            #mixture_samples, mixture_labels, test_data, test_labels = util.createTrainTest(real_features, fake_features)
            mixture_samples = np.concatenate([real_features, fake_features])
            mixture_labels = np.concatenate([np.ones(real_features.shape[0]), np.zeros(fake_features.shape[0])])
            ll_curve = llc.getPRCurve(mixture_samples, mixture_labels, lambdas, [mean_vec, cov_matrix], [shifted_mean, cov_matrix])

            curve_dict[curve_method_names[0]][dim][val] = curve_histo
            curve_dict[curve_method_names[1]][dim][val] = ll_curve

        if save == True:
            util.savePickle(path, curve_dict)

    for method, method_dict in curve_dict.items():
        for dim, dim_dict in method_dict.items():
            plotting.plotPRCurve(dim_dict, f"Method name {method} and dim {dim}", label_info="Add noise ")

