import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import experiments_v2.helper_functions as util
import visualize as plotting

def scorePlots(score_dataframe, score_dataframe_extra):
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.title("Recall")
    plotting.plotBars(score_dataframe, score_dataframe_extra, "recall")
    plt.subplot(2, 1, 2)
    plt.title("Precision")
    plotting.plotBars(score_dataframe, score_dataframe_extra, "precision")
    plt.xlabel("k-val")

def plotData(real, fake, extra_points=np.array([])):
    plt.figure()
    plt.scatter(real[:, 0], real[:, 1], label="Real data", color="green")
    plt.scatter(fake[:, 0], fake[:, 1], label="Fake data", color="blue")
    if extra_points.shape[0] > 0:
        plt.scatter(extra_points[:, 0], extra_points[:, 1], label="Added fake outliers", color="red")
    plt.legend()

def doEval(real_features, fake_features, k_vals):
    distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = util.getDistanceMatrices(
        real_features, fake_features)
    row_data = []
    columns = ["k_val", "recall", "precision"]
    for k_val in k_vals:
        # Calculations
        boundaries_real = distance_matrix_real[:, k_val]
        boundaries_fake = distance_matrix_fake[:, k_val]
        precision, recall, density, coverage = util.getScores(distance_matrix_pairs, boundaries_fake,
                                                              boundaries_real, k_val)
        row = [k_val, recall, coverage]
        row_data.append(row)

    dataframe = pd.DataFrame(columns=columns, data=row_data)
    return dataframe

def doTest():
    dimension = 2
    mean_real = np.zeros(dimension)
    cov_real = np.eye(dimension)
    cov_fake = np.eye(dimension)*0.01
    real_samples = 1000
    k_vals = util.getParams(real_samples)
    real_features = np.random.multivariate_normal(mean_real, cov_real, size=real_samples)
    fake_features = np.random.multivariate_normal(mean_real, cov_fake, size=real_samples)
    score_dataframe = doEval(real_features, fake_features, k_vals)
    corner_value=2
    random_distribution = np.array([[-corner_value,-corner_value], [-corner_value, corner_value], [corner_value,-corner_value], [corner_value,corner_value]])
    fake_extra = np.concatenate((fake_features,random_distribution), axis=0)
    score_dataframe_extra = doEval(real_features, fake_extra, k_vals)
    plotData(real_features, fake_features, random_distribution)
    scorePlots(score_dataframe, score_dataframe_extra)



doTest()
plt.show()