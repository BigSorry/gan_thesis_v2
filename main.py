import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import experiments_v2.fake_diverse as experiment_diverse
import experiments_v2.coverage_test as experiment3
import experiments_v2.uniform_test as experiment_uniform
import experiments_v2.helper_functions as helper
import visualize as plotting

def saveData(saved_data, real_data, fake_data, k_val):
    saved_data["real_data"].append(real_data)
    saved_data["fake_data"].append(fake_data)
    saved_data["k_val"].append(k_val)
    boundaries_real, boundaries_fake, distance_matrix_pairs = helper.getBoundaries(real_data, fake_data, k_val)
    max_boundary = max(np.max(boundaries_real), np.max(boundaries_fake))
    saved_data["max_boundary"].append(max_boundary)

def runFakeDiverse():
    iters = 1
    fake_samples = 100
    real_samples = 100
    radius = [0.1, 0.5, 1, 2, 8, 10, 16]
    #radius = [1, 2]
    k_vals = [1, 2, 3, 4, 8, 16, 25, 49]
    #k_vals = [1, 4]
    recall_tabular_data = np.zeros((len(radius), len(k_vals)))
    coverage_tabular_data = np.zeros((len(radius), len(k_vals)))
    saved_data = {"real_data": [], "fake_data": [], "k_val": [], "max_boundary": [], "titles": []}
    for column_id, k_val in enumerate(k_vals):
        for row_id, r in enumerate(radius):
            # Fake by circle sampling
            #real_data, fake_data = experiment2.getData(real_samples, fake_samples, r)
            # Sample only Gaussians
            real_data, fake_data = experiment_diverse.getDataNew(real_samples, fake_samples, r)
            for i in range(iters):
                precision, recall, density, coverage = experiment_diverse.constantRadii(real_data, fake_data,  k_val)
                recall_tabular_data[row_id, column_id] = recall
                coverage_tabular_data[row_id, column_id] = coverage
                #result_dict[r].append(scores)
                if i == 0:
                    saveData(saved_data, real_data, fake_data, k_val)
                    saved_data["titles"].append(f"Radius fake samples is {r}")

    rows = [f"K={k_val}" for k_val in k_vals]
    # Lambda char
    columns = [f"\u03BB={rad}" for rad in radius]
    plotting.saveHeatMap(recall_tabular_data, rows, columns,
                         save=True, save_path="./fig_v2/heatmaps/recall.png")
    plotting.saveHeatMap(coverage_tabular_data, rows, columns,
                         save=True, save_path="./fig_v2/heatmaps/coverage.png")
    plotting.plotInterface(saved_data, save=True, save_path="./fig_v2/recall/")
    #plotting.showScores(result_dict, save=True, save_path="./fig_v2/recall/")

def runFourClusters():
    iters = 10
    fake_samples = 40
    real_samples = 40
    k_vals = [1, 2, 4, 6, 8, 10, 12, 14, 16]
    result_dict = {k_val: [] for k_val in k_vals}
    saved_data = {"real_data": [], "fake_data": [], "k_val": [], "max_boundary": [], "titles": []}
    for i in range(iters):
        real_data, fake_data = experiment3.getData(real_samples, fake_samples)
        for k_val in k_vals:
            scores = experiment3.experimentCoverage(real_data, fake_data, k_val)
            result_dict[k_val].append(scores)
            if 1==1 and i == 0:
                saveData(saved_data, real_data, fake_data, k_val)
                saved_data["titles"].append(f"K value is {k_val}")

    plotting.plotInterface(saved_data, save=True, save_path="./fig_v2/coverage/")
    plotting.showScores(result_dict, save=True, save_path="./fig_v2/coverage/")

def doModeCollapse():
    iters = 1
    fake_samples = 100
    real_samples = 100
    variances = [0.01, 0.1, 0.25, 0.5, 0.75, 1]
    variances = [0.1, 0.25, 1, 2, 4]
    k_vals = [4]
    dimensions = [2, 4, 8, 16, 32, 64, 128]
    columns = ["k_val", "variance", "dimension", "precision", "recall", "density", "coverage"]
    row_data = []
    data_dict = {}
    for k_id, k_val in enumerate(k_vals):
        for variance_id, variance in enumerate(variances):
            for dim_id, dim in enumerate(dimensions):
                real_data, fake_data = experiment_diverse.getDataNew(real_samples, fake_samples, variance, dim)
                key = (variance, dim)
                if key not in data_dict:
                    data_dict[key] = [real_data, fake_data]
                for i in range(iters):
                    precision, recall, density, coverage = experiment_diverse.constantRadii(real_data, fake_data, k_val)
                    row = [k_val, variance, dim, precision, recall, density, coverage]
                    row_data.append(row)



    datafame = pd.DataFrame(columns=columns, data=row_data)
    plotting.plotDataFrame(datafame, "coverage")
    plotting.plotDataFrame(datafame, "recall")

def testExperiment():
    iters = 1
    total_samples = 40
    clusters = 4
    cluster_samples = total_samples // clusters
    k_vals = [1, 2, 4, 6, 8, 10, 12, 14, 16]
    result_dict = {k_val: [] for k_val in k_vals}
    saved_data = {"real_data": [], "fake_data": [], "k_val": [], "max_boundary": [], "titles": []}
    for i in range(iters):
        cluster_list = experiment3.getClusterData(cluster_samples)
        cluster_data = np.array(cluster_list).reshape(total_samples, 2)
        for drop in range(1, clusters):
            select_data = cluster_list[drop:]
            clusters_taken = len(select_data)
            cluster_data_taken = np.array(select_data).reshape(clusters_taken * cluster_samples, 2)
            for k_val in k_vals:
                scores = experiment3.experimentCoverage(cluster_data, cluster_data_taken, k_val)
                result_dict[k_val].append(scores)
                if 1 == 1:
                    saveData(saved_data, cluster_data, cluster_data_taken, k_val)
                    saved_data["titles"].append(f"K value is {k_val} and drop {drop}")

        plotting.plotInterface(saved_data, save=True, save_path="./fig_v2/mode_drop/")
        plotting.showScores(result_dict, save=True, save_path="./fig_v2/mode_drop/")

def createData(k_vals, variances, dimensions, real_samples, fake_samples, iters):
    columns = ["variance", "dimension", "boundaries_real", "boundaries_fake",
              "max_distances_real", "max_distances_fake", "recall", "coverage"]
    row_data = []
    for k_id, k_val in enumerate(k_vals):
        for variance_id, variance in enumerate(variances):
            for dim_id, dim in enumerate(dimensions):
                real_data, fake_data = experiment_diverse.getDataNew(real_samples, fake_samples, variance, dim)
                helper.checkMinMaxDistance(real_data, fake_data, k=1)
                for i in range(iters):
                    boundaries_real, boundaries_fake, distance_matrix_pairs = helper.getBoundaries(real_data, fake_data, k_val)
                    max_distances_real = np.max(distance_matrix_pairs, axis=1)
                    max_distances_fake = np.max(distance_matrix_pairs, axis=0)
                    precision, recall, density, coverage = experiment_diverse.constantRadii(real_data, fake_data, k_val)

                    row = [variance, dim, boundaries_real, boundaries_fake,
                           max_distances_real, max_distances_fake, recall, coverage]
                    row_data.append(row)

    datafame = pd.DataFrame(columns=columns, data=row_data)
    return datafame

def checkDistances(k_vals, variances, dimensions, real_samples, fake_samples, iters):
    columns = ["k_val", "variance", "dimension", "avg_difference_real", "avg_difference_fake",  "recall", "coverage"]
    row_data = []
    for k_id, k_val in enumerate(k_vals):
        for variance_id, variance in enumerate(variances):
            for dim_id, dim in enumerate(dimensions):
                all_ones = np.ones(real_samples)
                all_zeros = np.zeros(real_samples)
                real_data, fake_data = experiment_diverse.getDataNew(real_samples, fake_samples, variance, dim)
                min_distance_real, max_distances_real, min_distances_fake, max_distances_fake = helper.checkMinMaxDistance(real_data, fake_data, k=k_val)
                precision, recall, density, coverage = experiment_diverse.constantRadii(real_data, fake_data, k_val)
                distance_div = max_distances_real / min_distance_real
                differences_real = np.mean(np.abs(all_ones - distance_div))

                distance_div = max_distances_fake / min_distances_fake
                differences_fake = np.mean(np.abs(all_ones - distance_div))

                row_data.append([k_val, variance, dim, differences_real, differences_fake, recall, coverage])

    datafame = pd.DataFrame(columns=columns, data=row_data)

    return datafame

from sklearn.utils import shuffle
def distanceCheck():
    iters = 1
    fake_samples = 1000
    real_samples = 1000
    variances = [0.1,  1, 10]
    k_vals = [1, 4]
    dimensions = [2, 8, 16]
    dataframe_result = checkDistances(k_vals, variances, dimensions, real_samples, fake_samples, iters)
    plt.show()

    variances = dataframe_result["variance"].unique()
    k_vals = dataframe_result["k_val"].unique()
    for var in variances:
        for k_val in k_vals:
            select_data = dataframe_result.loc[(dataframe_result["variance"] == var) & (dataframe_result["k_val"] == k_val), :]
            recall = select_data["recall"].values
            coverage = select_data["coverage"].values
            dimensions = select_data["dimension"].values
            differences_real = select_data["avg_difference_real"].values
            differences_fake = select_data["avg_difference_fake"].values
            plt.figure()
            plt.suptitle(f"Lambda is {var} and k is {k_val}")
            plt.subplot(1, 2, 1)
            plt.plot(dimensions, recall, label="recall", color="blue")
            plt.plot(dimensions, coverage, label="coverage", color="red")
            plt.ylim([0, 1.1])
            plt.xlabel("dimension")
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(dimensions, differences_real, label="Mean absolute difference min-max distance real", color="red")
            plt.plot(dimensions, differences_fake, label="Mean absolute difference min-max distance fake", color="blue")
            plt.xlabel("dimension")
            plt.ylabel("Mean absolute difference")
            plt.legend()


def runAll():
    # runFakeDiverse()
    # runFourClusters()
    #doModeCollapse()
    #distanceCheck()
    #plt.show()
    helper.doVolumeExperiment()
    plt.show()


runAll()
