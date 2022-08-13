import numpy as np
import experiment_utils as utils
import visualize as plotting
import metrics.metrics as mtr
import metrics.likelihood_classifier as llc

def doOutlier(sample_count, dims, metric_params, outlier_ratio):
    curve_method_names, lambdas = utils.prepData()
    curve_dict = {name: {} for name in curve_method_names}
    for dim in dims:
        mean_vec = np.zeros(dim)
        cov_matrix = np.eye(dim)
        real_features = np.random.multivariate_normal(mean_vec, cov_matrix, size=sample_count)
        fake_features = np.random.multivariate_normal(mean_vec, cov_matrix, size=int(sample_count*(1-outlier_ratio)))
        cov_matrix2 = np.eye(dim) * 7
        fake_features_outliers = np.random.multivariate_normal(mean_vec, cov_matrix2, size=int(sample_count*outlier_ratio))
        fake_features = np.concatenate([fake_features, fake_features_outliers])

        pr_score, curve_histo, cluster_labels = mtr.getHistoPR(real_features, fake_features, cluster_param=metric_params[0])
        # Likelihood classifier
        #mixture_samples, mixture_labels, test_data, test_labels = util.createTrainTest(real_features, fake_features)
        mixture_samples = np.concatenate([real_features, fake_features])
        mixture_labels = np.concatenate([np.ones(real_features.shape[0]), np.zeros(fake_features.shape[0])])
        ll_curve = llc.getPRCurve(mixture_samples, mixture_labels, lambdas, [mean_vec, cov_matrix], [mean_vec, cov_matrix2])

        curve_dict[curve_method_names[0]][dim] = curve_histo
        curve_dict[curve_method_names[1]][dim] = ll_curve

    for method, dim_dict in curve_dict.items():
        plotting.plotPRCurve(dim_dict, f"Method name {method} and outlier ratio {outlier_ratio}", label_info="dim")