import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visualize as plotting
import experiments_v2.helper_functions as helper
import experiments_v2.fake_diverse as experiment_diverse

def checkDistances(k_vals, variances, dimensions, real_samples,
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
def distanceCheck(save=False):
    iters = 1
    fake_samples = 1000
    real_samples = 1000
    variances = [1]
    k_vals = [1, 16, 64, 501]
    k_vals = [20]
    dimensions = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    dimensions = [2, 64, 512, 1024]
    if save:
        dataframe_result = checkDistances(k_vals, variances, dimensions, real_samples, fake_samples, iters, save=True)
    else:
        dataframe_result = helper.readPickle("./pickle_data/dimension_effect.pickle")

    scores = ["recall", "coverage", "precision", "density"]
    scores = ["recall", "coverage"]
    show_distances = False
    for k in k_vals:
        print(k)
        sel_data = dataframe_result.loc[dataframe_result["k_val"] == k, :]
        grouped_data = sel_data.groupby(["dimension"])
        scores_mean = grouped_data[scores].mean()
        scores_std = grouped_data[scores].std()
        variance = (sel_data["variance"].unique())[0]
        title_text = f"K is {k} with lambda {variance}"
        plotting.plotScoreErrorbars(scores, scores_mean, scores_std, title_text)

        if show_distances:
            distances = ["avg_difference_real", "avg_difference_fake"]
            distances_mean = grouped_data[distances].mean()
            distances_std = grouped_data[distances].std()
            plotting.plotDistances(distances_mean, distances_std)


def showDimensions():
    iters = 1
    fake_samples = 1000
    real_samples = 1000
    variance = 1
    k_vals = [64]
    dim = 2
    dim = 512
    real_data, fake_data = experiment_diverse.getDataNew(real_samples, fake_samples,
                                                         variance, dim)
    mean = np.zeros(dim)
    cov = np.eye(dim)*10
    noise_vector = np.random.multivariate_normal(mean, cov, size=real_samples)
    real_data2 = real_data + noise_vector
    fake_data2 = fake_data + noise_vector
    scores = ["recall", "coverage"]
    data_dict = {"real_data": [], "fake_data": [], "k_val": [], "titles": []}
    for k in k_vals:
        boundaries_real, boundaries_fake, distance_matrix_pairs = helper.getBoundaries(real_data, fake_data, k)
        precision, recall, density, coverage = experiment_diverse.constantRadii(real_data, fake_data, k)
        real_sum = helper.sumBoundaries(boundaries_real)
        fake_sum = helper.sumBoundaries(boundaries_fake)
        print(coverage, recall)
        print(real_sum, fake_sum)
        min_distance_real, max_distances_real, min_distances_fake, max_distances_fake = helper.checkMinMaxDistance(
            real_data, fake_data, k=k)






def runAll():
    distanceCheck(save=True)
    showDimensions()
    plt.show()

runAll()