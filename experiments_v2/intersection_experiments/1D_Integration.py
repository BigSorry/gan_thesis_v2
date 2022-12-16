import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visualize as plotting
import experiments_v2.helper_functions as util
import seaborn as sns
from scipy.stats import wasserstein_distance

def doGrouping(value):
    if value < 0.2:
        return ">0.2"
    elif value < 0.4:
        return "0.2-0.4"
    elif value < 0.6:
        return "0.4-0.6"
    elif value < 0.8:
        return "0.6-0.8"
    else:
        return "<0.8"

def plotScores(dataframe, score_names):
    selected_data = dataframe.copy()
    selected_data["intersection_ratio"] = selected_data["intersection_ratio"].round(2)
    for score_name in score_names:
        plt.figure()
        plt.title(score_name)
        sns.boxplot(data=selected_data, x="intersection_ratio", y=score_name)
        plt.xlabel("Intersection ratio 2D Uniform = fake Area / real Area")
        plt.xticks(rotation=90)
        plt.ylabel("Score")

def createDistributions(params, sample_size, dimension, iters):
    functions = {}
    for param in params:
        for i in range(iters):
            samples = np.random.exponential(param, size=(sample_size, dimension))
            functions[-param, param, i] = samples

    return functions

def getMax(param, dimension):
    quantile = -np.log(1-0.99)/param
    return quantile

def getAreaCalc(base_param, other_param, x=10):
    threshold_base = getMax(base_param, dimension=2)
    threshold_other = getMax(other_param, dimension=2)

    take_max = max(threshold_base, threshold_other)
    base_area = 1 - np.exp(-(base_param)*take_max)
    other_area = 1 - np.exp(-(other_param)*take_max)

    return other_area / base_area

def getWassersteinDistance(data, other_data, dimensions=1):
    distance = wasserstein_distance(data[:, 0], other_data[:, 0])

    return distance

def getKlDivergence(base_param, compare_param):
    return  np.log(base_param) - np.log(compare_param) + (compare_param / base_param) - 1
def checkDistance(base_param, other_param):
    first_part = (-np.exp(-other_param) - 1) / other_param
    second_part = (-np.exp(-base_param) - 1) / base_param
    return first_part + second_part
def getDataframe(distribution_dict, k_vals):
    columns = ["key", "k_val", "intersection_ratio", "kl_score", "kl_reverse_score",
               "precision", "recall", "density", "coverage"]
    row_data = []
    for base_param, base_data in distribution_dict.items():
        high_param = base_param[1]
        for other_key, other_data in distribution_dict.items():
            other_high_param = other_key[1]
            max_element = max(np.max(base_data), np.max(other_data))
            intersect_ratio = getWassersteinDistance(base_data, other_data)
            intersect_ratio = getKlDivergence(high_param, other_high_param)
            kl_score = getKlDivergence(high_param, other_high_param)
            kl_reverse_score = getKlDivergence(other_high_param, high_param)
            distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = util.getDistanceMatrices(base_data, other_data)
            for k_val in k_vals:
                boundaries_real = distance_matrix_real[:, k_val]
                boundaries_fake = distance_matrix_fake[:, k_val]
                precision, recall, density, coverage = util.getScores(distance_matrix_pairs, boundaries_fake,
                                                              boundaries_real, k_val)
                row = [(high_param, other_high_param), k_val, intersect_ratio, kl_score, kl_reverse_score,
                       precision, recall, density, coverage]
                row_data.append(row)

    dataframe = pd.DataFrame(columns=columns, data=row_data)
    return dataframe

def showData(data_dict):
    for key, data in data_dict.items():
        plt.figure()
        plt.title(key)
        plt.scatter(data[:, 0], data[:, 1])
        break

def doExperiment(params):
    sample_size=1000
    dimension=2
    iters = 2
    k_vals = util.getParams(sample_size)
    k_vals = [1, 2, 4, 8, 16, 32, 64]
    distribution_dict = createDistributions(params, sample_size, dimension, iters)
    showData(distribution_dict)
    dataframe = getDataframe(distribution_dict, k_vals)
    corrs = dataframe.corr()
    score_names = ["precision", "recall", "density", "coverage"]
    plotScores(dataframe, score_names)


params = np.arange(0.1, 1.1, step=.2)
doExperiment(params)


#plot(score_dict, intersection_area, density_difference_intersection)
plt.show()