import numpy as np
from scipy import integrate
from sklearn.metrics import pairwise_distances
from scipy.stats import multivariate_normal

def getDistanceMatrices(real_features, fake_features):
    distance_matrix_real = pairwise_distances(real_features, real_features, metric='euclidean')
    distance_matrix_fake = pairwise_distances(fake_features, fake_features, metric='euclidean')
    distance_matrix_pairs = pairwise_distances(real_features, fake_features, metric='euclidean')
    distance_matrix_real = np.sort(distance_matrix_real, axis=1)
    distance_matrix_fake = np.sort(distance_matrix_fake, axis=1)

    return distance_matrix_real, distance_matrix_fake, distance_matrix_pairs

def getGaussianDimension(sample_size, dimension, transform_dimensions, lambda_factors):
    mean_vec = np.zeros(dimension)
    cov_ref = np.eye(dimension) * lambda_factors[0]
    cov_scaled = np.eye(dimension)
    index_transformed = np.random.choice(cov_scaled.shape[0], transform_dimensions, replace=False)
    for index in index_transformed:
        cov_scaled[index, index] = lambda_factors[1]
    reference_distribution = np.random.multivariate_normal(mean_vec, cov_ref, sample_size)
    scaled_distributions = np.random.multivariate_normal(mean_vec, cov_scaled, sample_size)

    return reference_distribution, scaled_distributions

def getStats(curve, metric_points):
    points_above = []
    nearest_distance = []
    for point in metric_points:
        # Clipping for density score
        point = np.clip(point, 0, 1)
        distances = point - curve
        l1_distances = np.sum(np.abs(distances), axis=1)
        l1_distances_sorted = np.sort(l1_distances)
        nearest_l1_distance = l1_distances_sorted[0]
        nearest_distance.append(nearest_l1_distance)
        row_check = (distances[:, 0] >= 0) & (distances[:, 1] >= 0)
        above_true = np.sum(row_check) > 0
        points_above.append(above_true)

    return np.array(points_above), np.array(nearest_distance)

def getPRLambdas(angle_count = 50):
    epsilon = 1e-10
    angles = np.linspace(epsilon, np.pi / 2 - epsilon, num=angle_count)
    lambdas = np.tan(angles)

    return lambdas

def getScores(truth, predictions):
    # Null hypothesis mixture data is from real
    # FPR false rejection null
    # FNR failure to reject null
    real_label = 1
    fake_label = 0
    epsilon = 1e-10
    real_samples = np.sum(truth == real_label)
    fake_samples = np.sum(truth == fake_label)
    fp =  np.sum((predictions==fake_label) & (truth==real_label))
    fn =  np.sum((predictions==real_label) & (truth==fake_label))

    fpr = fp / (real_samples+epsilon)
    fnr = fn / (fake_samples+epsilon)

    return fpr, fnr

def getDensities(real_data, fake_data, distribution_parameters, method_name="gaussian", norm=True):
    mixture_data = np.concatenate([real_data, fake_data])
    dimension = mixture_data.shape[1]
    if method_name == "gaussian":
        return multiGaus(mixture_data, dimension, distribution_parameters, norm)
    # elif method_name == "exponential":
    #     return multiExponential(mixture_data, dimension, distribution_parameters)

def multiGaus(mixture_data, dimension, scale_params, norm):
    mean_vec = np.zeros(dimension)
    cov_real = np.eye(dimension) * scale_params[0]
    cov_fake = np.eye(dimension) * scale_params[1]
    densities_real = multivariate_normal.pdf(mixture_data, mean=mean_vec, cov=cov_real)
    densities_fake = multivariate_normal.pdf(mixture_data, mean=mean_vec, cov=cov_fake)
    if norm:
        densities_real = densities_real / np.sum(densities_real)
        densities_fake = densities_fake / np.sum(densities_fake)

    return densities_real, densities_fake

def getCurveClassifier(distribution_name, real_data, fake_data, scale_factors):
    lambdas = getPRLambdas(angle_count=1000)
    densities_real, densities_fake = getDensities(real_data, fake_data, scale_factors,
                                                         method_name=distribution_name, norm=False)
    curve_pairs = []
    for scale in lambdas:
        predictions = (densities_real*scale >= densities_fake).astype(int)
        truth_labels = np.concatenate([np.ones(real_data.shape[0]), np.zeros(fake_data.shape[0])])
        fpr, fnr = getScores(truth_labels, predictions)
        precision_class = scale*fpr + fnr
        recall_class = precision_class / scale
        curve_pairs.append([np.clip(precision_class,0,1), np.clip(recall_class,0,1)])

    return np.array(curve_pairs)

def getEvaluationPairsDF(iters, k_vals, sample_size, dimension, dimension_transformed, ratios, real_scaling=False):
    base_value = 1
    row_info = []
    scaling_mode = "real_scaled" if real_scaling else "fale_scaled"
    for iter in range(iters):
        for ratio_index, ratio in enumerate(ratios):
            scale = base_value*ratio
            lambda_factors = [base_value, scale]
            reference_distribution, scaled_distribution = getGaussianDimension(sample_size, dimension, dimension_transformed, lambda_factors)
            if real_scaling:
                lambda_factors = [scale, base_value]
                distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = getDistanceMatrices(scaled_distribution, reference_distribution)
                curve_classifier = getCurveClassifier("gaussian", scaled_distribution, reference_distribution, lambda_factors)
            else:
                distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = getDistanceMatrices(reference_distribution, scaled_distribution)
                curve_classifier = getCurveClassifier("gaussian", reference_distribution, scaled_distribution, lambda_factors)

            auc = integrate.trapz(np.round(curve_classifier[:, 1], 2), np.round(curve_classifier[:, 0], 2))
            pr_pairs, dc_pairs = getKNN(distance_matrix_real, distance_matrix_fake, distance_matrix_pairs, k_vals)
            pr_aboves, pr_nearest_distances = getStats(curve_classifier, pr_pairs)
            dc_aboves, dc_nearest_distances = getStats(curve_classifier, dc_pairs)

            for index, k_val in enumerate(k_vals):
                pr_score = pr_pairs[index, :]
                dc_score = dc_pairs[index, :]
                pr_distance = pr_nearest_distances[index]
                dc_distance = dc_nearest_distances[index]
                pr_above = pr_aboves[index]
                dc_above = dc_aboves[index]
                pr_row = [ratio_index, "pr", scaling_mode, iter, dimension, dimension_transformed, auc, k_val, pr_distance, pr_score[0], pr_score[1], pr_above]
                dc_row = [ratio_index, "dc", scaling_mode, iter, dimension, dimension_transformed, auc, k_val, dc_distance, dc_score[0], dc_score[1], dc_above]
                row_info.append(pr_row)
                row_info.append(dc_row)

    return row_info

def getEvaluationPair(iters, k_vals, sample_size, dimension, dimension_transformed,
                       ratios, real_scaling=False, return_curve=False):
    base_value = 1
    result_dict = {}
    for iter in range(iters):
        for ratio_index, ratio in enumerate(ratios):
            scale = base_value*ratio
            lambda_factors = [base_value, scale]
            reference_distribution, scaled_distribution = getGaussianDimension(sample_size, dimension, dimension_transformed, lambda_factors)
            if real_scaling:
                lambda_factors = [scale, base_value]
                distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = getDistanceMatrices(scaled_distribution, reference_distribution)
                curve_classifier = getCurveClassifier("gaussian", scaled_distribution, reference_distribution, lambda_factors)
            else:
                distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = getDistanceMatrices(reference_distribution, scaled_distribution)
                curve_classifier = getCurveClassifier("gaussian", reference_distribution, scaled_distribution, lambda_factors)

            auc = integrate.trapz(np.round(curve_classifier[:, 1], 2), np.round(curve_classifier[:, 0], 2))
            pr_pairs, dc_pairs = getKNN(distance_matrix_real, distance_matrix_fake, distance_matrix_pairs, k_vals)
            pr_aboves, pr_nearest_distances = getStats(curve_classifier, pr_pairs)
            dc_aboves, dc_nearest_distances = getStats(curve_classifier, dc_pairs)

            experiment_key = (iter, dimension, ratio, auc)

            result_dict[experiment_key] = {}
            result_dict[experiment_key]["pr_pairs"] = pr_pairs
            result_dict[experiment_key]["pr_nearest_distances"] = pr_nearest_distances
            result_dict[experiment_key]["dc_pairs"] = dc_pairs
            result_dict[experiment_key]["dc_nearest_distances"] = dc_nearest_distances
            if return_curve:
                result_dict[experiment_key]["curve"] = curve_classifier

    return result_dict

def getKNN(distance_real, distance_fake, distance_pairs, k_vals):
    pr_pairs = np.zeros((len(k_vals), 2))
    dc_pairs = np.zeros((len(k_vals), 2))
    for i, k_val in enumerate(k_vals):
        boundaries_real = distance_real[:, k_val]
        boundaries_fake = distance_fake[:, k_val]
        precision, recall, density, coverage = getMetricScores(distance_pairs, boundaries_fake, boundaries_real, k_val)
        pr_pairs[i, :] = [precision, recall]
        dc_pairs[i, :] = [density, coverage]

    return [pr_pairs, dc_pairs]

def getMetricScores(distance_matrix_pairs, boundaries_fake, boundaries_real, k_val):
    precision = (distance_matrix_pairs < np.expand_dims(boundaries_real, axis=1)).any(axis=0).mean()
    recall = (distance_matrix_pairs < np.expand_dims(boundaries_fake, axis=0)).any(axis=1).mean()
    # density coverage
    density = (1. / float(k_val)) * (distance_matrix_pairs < np.expand_dims(boundaries_real, axis=1)).sum(
        axis=0).mean()
    coverage = (distance_matrix_pairs.min(axis=1) < boundaries_real).mean()

    return [precision, recall, density, coverage]

