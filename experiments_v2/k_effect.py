import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visualize as plotting
import experiments_v2.helper_functions as helper
import experiments_v2.fake_diverse as experiment_diverse

def makeDataFrame(k_vals, variances, dimensions, real_samples,
                   fake_samples, iters, save=False):
    columns = ["iter_id", "k_val", "variance", "dimension",
               "avg_difference_real", "avg_difference_fake",  "recall", "coverage"]
    row_data = []
    for i in range(iters):
        for k_id, k_val in enumerate(k_vals):
            for variance_id, variance in enumerate(variances):
                for dim_id, dim in enumerate(dimensions):
                    all_ones = np.ones(real_samples)
                    #all_zeros = np.zeros(real_samples)
                    real_data, fake_data = experiment_diverse.getDataNew(real_samples, fake_samples, variance, dim)
                    min_distance_real, max_distances_real, min_distances_fake, max_distances_fake = helper.checkMinMaxDistance(real_data, fake_data, k=k_val)
                    precision, recall, density, coverage = experiment_diverse.constantRadii(real_data, fake_data, k_val)
                    distance_div = max_distances_real / min_distance_real
                    differences_real = np.mean(np.abs(all_ones - distance_div))

                    distance_div = max_distances_fake / min_distances_fake
                    differences_fake = np.mean(np.abs(all_ones - distance_div))


                    row_data.append([i, k_val, variance, dim, differences_real, differences_fake, recall, coverage])

    datafame = pd.DataFrame(columns=columns, data=row_data)
    if save:
        helper.savePickle("./pickle_data/dimension_effect.pickle", datafame)

    return datafame

from sklearn.utils import shuffle
def doExperiment():
    iters = 1
    fake_samples = 2000
    real_samples = 2000
    variance = .5
    k_vals = [1, 2, 4, 8, 16, 32, 100, 250, 500, 999, 1500, 1999]
    dim = 2
    real_data, fake_data = experiment_diverse.getDataNew(real_samples, fake_samples,
                                                         variance, dim)
    distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = helper.getDistanceMatrices(real_data, fake_data)
    data_dict = {"real_data": [], "fake_data": [], "k_val": [], "titles": [], "recall":[], "coverage":[],
                 "area_real" :[], "area_fake":[]}
    for k in k_vals:
        boundaries_real = distance_matrix_real[:, k]
        boundaries_fake = distance_matrix_fake[:, k]
        precision, recall, density, coverage = helper.getScores(distance_matrix_pairs, boundaries_fake,
                                                                boundaries_real, k)
        area_real = np.mean(helper.getArea(boundaries_real))
        area_fake = np.mean(helper.getArea(boundaries_fake))
        data_dict["real_data"].append(real_data)
        data_dict["fake_data"].append(fake_data)
        data_dict["recall"].append(recall)
        data_dict["coverage"].append(coverage)
        data_dict["area_real"].append(area_real)
        data_dict["area_fake"].append(area_fake)
        data_dict["k_val"].append(k)
        data_dict["titles"].append(f"K is {k}")


    plt.figure()
    plt.title("Sum of all Areas")
    plt.plot(data_dict["k_val"], data_dict["area_real"], label="real")
    plt.plot(data_dict["k_val"], data_dict["area_fake"], label="fake")
    plt.legend()
    plt.figure()
    plt.title("Scores")
    plt.plot(data_dict["k_val"], data_dict["recall"], label="recall")
    plt.plot(data_dict["k_val"], data_dict["coverage"], label=" coverage")
    plt.legend()

    #plotting.plotInterface(data_dict, save=False, save_path="")

def runAll():
    doExperiment()
    plt.show()

runAll()