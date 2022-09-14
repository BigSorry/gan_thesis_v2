import numpy as np
from scipy.linalg import sqrtm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
import experiment_utils as util
import metrics.improved_precision_recall as ipr
import metrics.precision_recall_histogram as prh
import metrics.precision_recall_classifier as prc
import metrics.likelihood_classifier as llc
import metrics.diversity_coverage as dc
import metrics.improved_precision_recall as ipr

# TODO Refactoring with Metric objects
def getPRCurve(real_features, fake_features, method_name, params):
    curve = None
    labels = None
    pr_score = None
    if method_name == "kmeans":
        pr_score, curve, labels = getHistoPR(real_features, fake_features, params)
    elif "classifier" in method_name:
        if "likelihood_classifier" in method_name:
            pr_score, curve, labels = getClassifierLikelihoodPR(real_features, fake_features, params)
        else:
            pr_score, curve, labels = getClassifierPR(real_features, fake_features, params)
    elif "2d" in method_name:
        if method_name == "2d_density_coverage":
            pr_score = getDensityRecall(real_features, fake_features, params)
        elif method_name == "2d_knn":
            pr_score = getKNN(real_features, fake_features, params)

    return pr_score, curve, labels

def getDensityRecall(real_features, fake_features, params):
    k = params["k"]
    pr_score = dc.getDensityRecall(real_features, fake_features, k)

    return pr_score

def getKNN(real_features, fake_features, params):
    k = params["k"]
    pr_score = ipr.getPrecisionRecall(real_features, fake_features, k)

    return pr_score

def getClassifierLikelihoodPR(real_features, fake_features, params):
    angle_count = params["angles"]
    mean_vectors = params["mean_vectors"]
    covariance = params["covariance"]
    fake_mode_samples = params["fake_modes"]
    mode_weights_real = np.ones(mean_vectors.shape[0]) / mean_vectors.shape[0]
    mode_weights_fake = fake_mode_samples / np.sum(fake_mode_samples)
    real_params = [mean_vectors, covariance, mode_weights_real]
    fake_params = [mean_vectors, covariance, mode_weights_fake]
    lambdas = util.getLambdas(angle_count)
    mixture_samples = np.concatenate((real_features, fake_features))
    labels = np.concatenate((np.ones(real_features.shape[0]),
                                 np.zeros(fake_features.shape[0])))

    pr_curve = llc.getPRCurve(mixture_samples, labels, lambdas,
               real_params, fake_params)

    return None, pr_curve, None



def getClassifierPR(real_features, fake_features, params):
    threshold_count = params["threshold_count"]
    angle_count = params["angles"]
    classifier = params["classifier"]["object"]
    lambdas = util.getLambdas(angle_count)
    #beta_score = params["beta_score"]
    train, test, train_labels, test_labels = prc.createTrainTest(real_features, fake_features)
    curve, ths, prob_labels = prc.getPrecisionRecallCurve(train, train_labels, test, test_labels,
                                                          lambdas, threshold_count, classifier)

    pr_score = prh.prd_to_max_f_beta_pair(curve[:, 0], curve[:, 1], beta=1)

    return pr_score, curve, prob_labels

def getHistoPR(real_features, other_features, params):
    cluster_param = params["k_cluster"]
    angles = params["angles"]
    runs = params["kmeans_runs"]
    #beta_score = params["beta_score"]
    precision_vals, recall_vals, cluster_labels = prh.compute_prd_from_embedding(real_features, other_features,
                                                                                 num_clusters=cluster_param, num_angles=angles, num_runs=runs)

    pr_score = prh.prd_to_max_f_beta_pair(precision_vals, recall_vals, beta=8)
    curve = np.array((precision_vals, recall_vals)).T

    return pr_score, curve, cluster_labels

def fidScore(activation_real, activation_fake, cache=None):
    mean1 = np.mean(activation_real, axis=0) if cache is None else cache["mean"]
    mean2 = np.mean(activation_fake, axis=0)

    cov1 = np.cov(activation_real, rowvar=False) if cache is None else cache["cov"]
    cov2 = np.cov(activation_fake, rowvar=False)
    first_term = np.linalg.norm(mean1 - mean2)
    cov_product = sqrtm(cov1.dot(cov2))
    if np.iscomplexobj(cov_product):
        cov_product = cov_product.real
    second_term = np.trace(cov1) + np.trace(cov2) - 2*np.trace(cov_product)

    return first_term + second_term

def fidScoreParams(real_params, fake_params):
    mean1 = real_params[0]
    mean2 = fake_params[0]

    cov1 = real_params[1]
    cov2 = fake_params[1]
    first_term = np.linalg.norm(mean1 - mean2)
    cov_product = sqrtm(cov1.dot(cov2))
    if np.iscomplexobj(cov_product):
        cov_product = cov_product.real
    second_term = np.trace(cov1) + np.trace(cov2) - 2*np.trace(cov_product)

    return first_term + second_term


def assertDims(x, y):
    assert x.shape[0] == y.shape[0], ("Row dimensions must be the same")
    assert x.shape[1] == y.shape[1], ("Column dimensions must be the same")

# reference
# https://github.com/djsutherland/opt-mmd/blob/master/two_sample/mmd.py
# Zero real_data == fake_data
def rbfMMD(activation_real, activation_fake, sigma=1):
    assertDims(activation_real, activation_fake)
    gamma = 1 / (2 * sigma ** 2)

    xx = np.dot(activation_real, activation_real.T)
    xy = np.dot(activation_real, activation_fake.T)
    yy = np.dot(activation_fake, activation_fake.T)

    x_norms = np.diag(xx)
    y_norms = np.diag(yy)

    k_xy = np.exp(-gamma * (
            -2 * xy + x_norms[:, np.newaxis] + y_norms[np.newaxis, :]))
    k_xx = np.exp(-gamma * (
            -2 * xx + x_norms[:, np.newaxis] + x_norms[np.newaxis, :]))
    k_yy = np.exp(-gamma * (
            -2 * yy + y_norms[:, np.newaxis] + y_norms[np.newaxis, :]))

    m = k_xx.shape[0]
    n = k_yy.shape[0]
    mmd = ((np.sum(k_xx) - m) / (m * (m - 1))
            + (np.sum(k_yy) - n) / (n * (n - 1))
            - 2 * np.mean(k_xy))
    return mmd

def nnScore(activation_real, activation_fake):
    assertDims(activation_real, activation_fake)
    neigh = KNeighborsClassifier(n_neighbors=1)
    real_labels = [1 for i in range(activation_real.shape[0])]
    fake_labels = [-1 for i in range(activation_fake.shape[0])]
    X = np.concatenate([activation_real, activation_fake])
    y = np.array(real_labels + fake_labels)
    loo = LeaveOneOut()
    score = 0
    for train, test in loo.split(X):
        train_split, train_labels = X[train], y[train]
        neigh.fit(train_split, train_labels)
        test_split, test_label = X[test], y[test][0]
        prediction = neigh.predict(test_split)[0]
        score += 1 if prediction == test_label else 0

    return score / X.shape[0]

