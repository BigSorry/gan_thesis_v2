import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visualize as plotting
import experiments_v2.helper_functions as util
import seaborn as sns

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
            samples = np.random.uniform(-param, param, size=(sample_size, dimension))
            functions[-param, param, i] = samples

    return functions

def getAreaCalc(base_param, other_param):
    base_area = (base_param - - base_param) ** 2
    other_area = (other_param - - other_param) ** 2
    density_difference = np.abs((1 / base_area) - (1 / other_area))
    intersect_ratio = other_area / base_area

    return density_difference, intersect_ratio

def getDataframe(distribution_dict, k_vals):
    columns = ["key", "k_val", "intersection_ratio", "density_diff",
               "precision", "recall", "density", "coverage"]
    row_data = []
    for base_param, base_data in distribution_dict.items():
        high_param = base_param[1]
        for other_key, other_data in distribution_dict.items():
            other_high_param = other_key[1]
            density_difference, intersect_ratio = getAreaCalc(high_param, other_high_param)
            distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = util.getDistanceMatrices(base_data, other_data)
            for k_val in k_vals:
                boundaries_real = distance_matrix_real[:, k_val]
                boundaries_fake = distance_matrix_fake[:, k_val]
                precision, recall, density, coverage = util.getScores(distance_matrix_pairs, boundaries_fake,
                                                              boundaries_real, k_val)
                row = [(high_param, other_high_param), k_val, intersect_ratio, density_difference,
                       precision, recall, density, coverage]
                row_data.append(row)

    dataframe = pd.DataFrame(columns=columns, data=row_data)
    return dataframe

def doCorr(dataframe, score_names):
    medians = {}
    stds = {}
    selected_data = dataframe.loc[dataframe["intersection_ratio"] <= 1, :]
    selected_data2 = dataframe.loc[dataframe["intersection_ratio"] > 1, :]
    corr = selected_data.corr()
    corr2 = selected_data2.corr()
    for score_name in score_names:
        medians[score_name] = dataframe[score_name].median()
        stds[score_name] = dataframe[score_name].std()

def doExperiment(params):
    sample_size=100
    dimension=2
    iters = 2
    k_vals = util.getParams(sample_size)
    print(params)
    print(k_vals)
    distribution_dict = createDistributions(params, sample_size, dimension, iters)
    dataframe = getDataframe(distribution_dict, k_vals)
    score_names = ["precision", "recall", "density", "coverage"]
    plotScores(dataframe, score_names)
    doCorr(dataframe, score_names)

params = np.arange(0.1, 1.1, step=0.2)
doExperiment(params)


#plot(score_dict, intersection_area, density_difference_intersection)
plt.show()