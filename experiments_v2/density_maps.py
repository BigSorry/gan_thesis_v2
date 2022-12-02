import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visualize as plotting
import experiments_v2.helper_functions as util
import seaborn as sns

def plotScores(dataframe, score_names):
    x = dataframe["intersection_ratio"].values
    y = dataframe["density_diff"].values
    for name in score_names:
        z = dataframe[name].values
        plt.figure()
        plt.title(name)
        plt.scatter(x, z)
        plt.xlim([0, 1])


def createDistributions(params, sample_size, dimension):
    functions = {}
    for param in params:
        samples = np.random.uniform(-param, param, size=(sample_size, dimension))
        functions[-param, param] = samples

    return functions

def getParams(sets,  sample_size, dimension):
    return

def getAreaCalc(base_param, other_param):
    base_area = (base_param - - base_param) ** 2
    other_area = (other_param - - other_param) ** 2
    density_difference = np.abs((1 / base_area) - (1 / other_area))
    intersect_ratio = other_area / base_area

    return density_difference, intersect_ratio

def getDataframe(distribution_dict):
    columns = ["key", "intersection_ratio", "density_diff",
               "precision", "recall", "density", "coverage"]
    row_data = []
    for base_param, base_data in distribution_dict.items():
        high_param = base_param[1]
        for other_key, other_data in distribution_dict.items():
            other_high_param = other_key[1]
            density_difference, intersect_ratio = getAreaCalc(high_param, other_high_param)

            precision, recall, density, coverage = getScores(base_data, other_data, k_val=7)
            row = [(high_param, other_high_param), intersect_ratio, density_difference, precision, recall, density, coverage]
            row_data.append(row)

    dataframe = pd.DataFrame(columns=columns, data=row_data)
    return dataframe

def getScores(real_features, fake_features, k_val):
    distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = util.getDistanceMatrices(
        real_features, fake_features)
    boundaries_real = distance_matrix_real[:, k_val]
    boundaries_fake = distance_matrix_fake[:, k_val]
    precision, recall, density, coverage = util.getScores(distance_matrix_pairs, boundaries_fake,
                                                          boundaries_real, k_val)

    return [precision, recall, density, coverage]


def doExperiment(params):
    sample_size=1000
    dimension=2

    print(params)
    distribution_dict = createDistributions(params, sample_size, dimension)
    dataframe = getDataframe(distribution_dict)
    score_names = ["precision", "recall", "density", "coverage"]
    plotScores(dataframe, score_names)

params = np.arange(0.1, 1.1, step=0.1)
doExperiment(params)


#plot(score_dict, intersection_area, density_difference_intersection)
plt.show()