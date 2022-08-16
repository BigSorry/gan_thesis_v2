import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visualize as plotting
import experiments_v2.helper_functions as helper
import experiments_v2.fake_diverse as experiment_diverse

def getDataframe(k_vals, distance_matrix_real, distance_matrix_fake, distance_matrix_pairs,
                  iters, save=False):
    columns = ["iter_id", "k_val", "area_real", "area_fake",  "recall", "coverage"]
    row_data = []
    for i in range(iters):
        for k_id, k_val in enumerate(k_vals):
            boundaries_real = distance_matrix_real[:, k_val]
            boundaries_fake = distance_matrix_fake[:, k_val]
            precision, recall, density, coverage = helper.getScores(distance_matrix_pairs, boundaries_fake,
                                                                    boundaries_real, k_val)
            area_real = np.sum(helper.getArea(boundaries_real))
            area_fake = np.sum(helper.getArea(boundaries_fake))


            row_data.append([i, k_val, area_real, area_fake, recall, coverage])

    datafame = pd.DataFrame(columns=columns, data=row_data)
    if save:
        helper.savePickle("./pickle_data/dimension_effect.pickle", datafame)

    return datafame

from sklearn.utils import shuffle
# TODO Refactoring once results are explained clearly
def doExperiment():
    # Set experiment params
    iters = 1
    samples_both = 1000
    fake_samples = samples_both
    real_samples = samples_both
    variance = 0.01
    k_vals = np.linspace(1, 999, 50, dtype=int)
    dimensions = [2, 4, 8, 12, 16, 25, 32, 512]
    dimensions = [2]
    # Prep dataframe
    for dim in dimensions:
        real_data, fake_data = experiment_diverse.getDataNew(real_samples, fake_samples,
                                                             variance, dim)
        distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = helper.getDistanceMatrices(real_data, fake_data)
        dataframe = getDataframe(k_vals, distance_matrix_real, distance_matrix_fake, distance_matrix_pairs, iters)
        dataframe["real_div_norm"] = (dataframe["area_real"] - dataframe["area_real"].min()) /\
                                     (dataframe["area_real"].max() - dataframe["area_real"].min())
        dataframe["fake_div_norm"] = (dataframe["area_fake"] - dataframe["area_fake"].min()) /\
                                     (dataframe["area_fake"].max() - dataframe["area_fake"].min())

        # Do plotting
        if dim == 2:
            plt.figure()
            plt.title(f"Sum of all Areas with dim is {dim}")
            plt.plot(dataframe["k_val"], dataframe["real_div_norm"], label="real")
            plt.plot(dataframe["k_val"], dataframe["fake_div_norm"], label="fake")
            plt.ylim([0, 1.1])
            plt.xscale('log')
            plt.legend()
            plt.figure()
            plt.title(f"Scores with dim is {dim}")
            plt.plot(dataframe["k_val"], dataframe["recall"], label="recall")
            plt.plot(dataframe["k_val"], dataframe["coverage"], label=" coverage")
            plt.ylim([0, 1.1])
            plt.xscale('log')
            plt.legend()
            for index, k in enumerate(k_vals):
                if index % 10 == 0:
                    data_dict = {"real_data":[real_data], "fake_data":[fake_data], "k_val":[k], "titles":[f"K-val is {k}"]}
                    plotting.plotInterface(data_dict, save=False, save_path="")

def runAll():
    doExperiment()
    plt.show()

runAll()