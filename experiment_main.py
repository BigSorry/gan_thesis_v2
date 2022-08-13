import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
import visualize as plotting
import experiment_utils as util
import metrics.precision_recall_histogram as prh
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from CreatorExperiment import CreateExperiment
from torchvision import datasets
from models.inceptionv3 import *
import torchvision.models as models

def getPlotTitle(experiment_id):
    text = ""
    if experiment_id == 1:
        text = "Mean shifted by "
    if experiment_id == 2:
        text = "Covariance diagonal zero percentage "
    if experiment_id == 3:
        text = "Concentration of first mode  "
    return text

def getName(id):
    name = ""
    if id == 1:
        name = "mean_shift"
    if id == 2:
        name = "covariance"
    if id == 3:
        name = "drop_gradually"
    return name

# Only relevant for precision and recall curve metrics
def doBetaExperiment(experiment_id, metric_names, sample_array, dim_array):
    experiment_name = getName(experiment_id)
    plot_title = getPlotTitle(experiment_id)
    path_result = f"./saved_pkl/{experiment_name}/"
    time_experiment = datetime.datetime.now().strftime("%d_%m_%Y_h%H_m%M_s%S")
    result_path = f"{path_result}result_test-{time_experiment}.pkl"
    all_results = {}
    runs = 1
    for i in range(runs):
        for metric_name in metric_names:
            metric_params = getParamDict(metric_name)
            #metric_params["beta_score"] = beta_score
            # Metric name change for classifier method
            if "classifier" in metric_name:
                classifier_name = metric_params["classifier"]["name"]
                metric_name = f"classifier_{classifier_name}"
            if metric_name not in all_results:
                all_results[metric_name] = {}

            experiment_object = CreateExperiment().getExperiment(experiment_id)
            experiment_object.setSamples(sample_array)
            experiment_object.setDimensions(dim_array)
            # Saves results
            result_dict = experiment_object.doExperiment(metric_name, metric_params)
            all_results[metric_name] = result_dict["experiments"]

    # Overwrite default beta calc
    beta_vals = np.arange(10)+1
    beta_vals = np.array([1, 2, 4, 6, 8, 10])
    beta_vals = np.around(np.concatenate([1/beta_vals, beta_vals[1:]]), decimals=2)
    beta_vals.sort()
    for metric_name, metric_results in all_results.items():
        for experiment_key, experiment_dict in metric_results.items():
            pr_curves = experiment_dict["curve"]
            experiment_dict["pr_score"] = {}
            experiment_dict["f_scores"] = {}
            for curve in pr_curves:
                precision_vals = curve[:, 0]
                recall_vals = curve[:, 1]
                for beta_score in beta_vals:
                    f_scores = prh._prd_to_f_beta(precision_vals, recall_vals, beta_score)
                    new_pr_score = prh.prd_to_max_f_beta_pair(precision_vals, recall_vals, beta_score)
                    experiment_dict["pr_score"][beta_score] = new_pr_score
                    experiment_dict["f_scores"][beta_score] = f_scores

    plotting.plotCurves(all_results, plot_title)
    #plotting.plotFScores(all_results, plot_title, save=True)

def doExperiment(experiment_id, metric_names, sample_array, dim_array):
    experiment_name = getName(experiment_id)
    plot_title = getPlotTitle(experiment_id)
    path_result = f"./saved_pkl/{experiment_name}/"
    time_experiment = datetime.datetime.now().strftime("%d_%m_%Y_h%H_m%M_s%S")
    result_path = f"{path_result}result_test-{time_experiment}.pkl"
    all_results = {}
    for metric_name in metric_names:
        metric_params = getParamDict(metric_name)
        if "classifier" in metric_name:
            classifier_name = metric_params["classifier"]["name"]
            metric_name = f"classifier_{classifier_name}"

        experiment_object = CreateExperiment().getExperiment(experiment_id)
        experiment_object.setSamples(sample_array)
        experiment_object.setDimensions(dim_array)
        # Saves results
        result_dict = experiment_object.doExperiment(metric_name, metric_params)
        all_results[metric_name] = result_dict["experiments"]

    plotting.plotAllResults(all_results, plot_title)

from sklearn.neighbors import KNeighborsClassifier
def getParamDict(method_name):
    params = {}
    if "2d" in method_name:
        params = {"k": 7}
    elif method_name == "kmeans":
        params = {"k_cluster": 20, "angles": 1001, "kmeans_runs": 5}
    elif "classifier" in method_name:
        clf = LogisticRegression()
        if "1" in method_name:
            clf = RandomForestClassifier()
        params = {"threshold_count": 500, "angles": 1001, "runs": 1}
        params["classifier"] = {"name": clf.__class__.__name__,
                                        "object": clf}
    return params

# TODO kmeans desnity
# TODO scaling distributions by removing furthest sample
def main():
    experiment_ids = [1, 2, 3, 4, 5]
    experiment_ids = [2]
    for experiment_id in experiment_ids:
        if 1==1:
            samples = [1000]
            dimensions = [100]
            method_names = ["2d_knn", "2d_density_coverage", "kmeans"]
            if experiment_id < 5:
                doExperiment(experiment_id, method_names, samples, dimensions)
            else:
                experimentActivations(experiment_id, method_names)

from sklearn.decomposition import PCA
def experimentActivations(experiment_id, metric_names):
    experiment_name = ""
    plot_title = ""
    path_result = f"./saved_pkl/{experiment_name}/"
    time_experiment = datetime.datetime.now().strftime("%d_%m_%Y_h%H_m%M_s%S")
    all_results = {}
    path= "./activations/activation_train_mnist.pkl"
    activation = util.readPickle(path)
    for metric_name in metric_names:
        metric_params = getParamDict(metric_name)
        if "classifier" in metric_name:
            classifier_name = metric_params["classifier"]["name"]
            metric_name = f"classifier_{classifier_name}"
        experiment_object = CreateExperiment().getExperiment(experiment_id)
        experiment_object.setActivations(activation)
        experiment_object.setSamples(0)
        experiment_object.setDimensions(0)
        # Saves results
        result_dict = experiment_object.doExperiment(metric_name, metric_params)
        all_results[metric_name] = result_dict["experiments"]

    plotting.plotAllResults(all_results, plot_title)
    #plotting.plotCurves(all_results, plot_title)







main()
plt.show()