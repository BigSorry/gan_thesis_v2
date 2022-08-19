import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visualize as plotting
import experiments_v2.helper_functions as helper
import experiments_v2.fake_diverse as experiment_diverse

def plotLines(var_x, var_y1, var_y2, text_info, save=False):
    # Do plotting
    plt.figure()
    plt.title(text_info["title_text"])
    plt.plot(var_x, var_y1, label=text_info["label1"], c="blue")
    plt.plot(var_x, var_y2, label=text_info["label2"], c="red")
    plt.xlabel("K value")
    plt.ylim([0, 1.1])
    plt.xscale('log')
    plt.legend()
    if save == True:
        plt.savefig(text_info["save_path"], bbox_inches="tight")

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


def normUnion(real, fake):
    combined = pd.concat([real, fake])
    real_normed = (real - combined.min()) / \
                                 (combined.max() - combined.min())
    fake_normed = (fake - combined.min()) / \
                                 (combined.max() - combined.min())

    return real_normed, fake_normed

def normSeparate(real, fake):
    real_normed = (real - real.min()) / \
                                 (real.max() - real.min())
    fake_normed = (fake - fake.min()) / \
                                 (fake.max() - fake.min())

    return real_normed, fake_normed

from sklearn.utils import shuffle
# TODO Refactoring once results are explained clearly
def doExperiment(savePlotLines):
    # Set experiment params
    iters = 1
    samples_both = 1000
    fake_samples = samples_both
    real_samples = samples_both
    variance = 0.1
    k_vals = np.linspace(1, 999, 50, dtype=int)
    dimensions = [2, 4, 8, 12, 16, 25, 32, 512]
    dimensions = [2, 4, 16, 32, 64]
    # Prep dataframe
    for dim in dimensions:
        real_data, fake_data = experiment_diverse.getDataNew(real_samples, fake_samples,
                                                             variance, dim)
        distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = helper.getDistanceMatrices(real_data, fake_data)
        dataframe = getDataframe(k_vals, distance_matrix_real, distance_matrix_fake, distance_matrix_pairs, iters)
        real_normed_un, fake_normed_un = normUnion(dataframe["area_real"], dataframe["area_fake"])
        real_normed_sep, fake_normed_sep = normSeparate(dataframe["area_real"], dataframe["area_fake"])

        text_info = {"label1":"Real data", "label2": "Fake data", "save_path":f"../fig_v2/k_effect/areas_un_dim{dim}.png",
                     "title_text": f"Union normalized sum of ball area's with dimension {dim}"}
        plotLines(dataframe["k_val"], real_normed_un,  fake_normed_un, text_info, save=savePlotLines)

        text_info = {"label1": "Real data", "label2": "Fake data",
                     "save_path": f"../fig_v2/k_effect/areas_sep_dim{dim}.png",
                     "title_text": f"Separately normalized sum of ball area's with dimension {dim}"}
        plotLines(dataframe["k_val"], real_normed_sep, fake_normed_sep, text_info, save=savePlotLines)

        text_info = {"label1": "Coverage", "label2": "Recall", "save_path":f"../fig_v2/k_effect/scores_dim{dim}.png",
                      "title_text": f"Metric scores with dimension {dim}"}
        print(dataframe["coverage"].mean(), dataframe["recall"].mean())
        plotLines(dataframe["k_val"], dataframe["coverage"], dataframe["recall"], text_info, save=True)
        # if dim == 2:
        #     for index, k in enumerate(k_vals):
        #         if index % 20 == 0 or k == 999:
        #             data_dict = {"real_data":[real_data], "fake_data":[fake_data], "k_val":[k], "titles":[f"K-val is {k}"]}
        #             plotting.plotInterface(data_dict, save=False, save_path="")

def runAll():
    savePlotLines = False
    doExperiment(savePlotLines)

    if savePlotLines is False:
        plt.show()

runAll()