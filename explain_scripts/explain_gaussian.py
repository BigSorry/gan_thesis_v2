import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import experiments_v2.helper_functions as util
import visualize as plotting

def plotData(real, fake):
    plt.figure()
    plt.scatter(real[:, 0], real[:, 1], label="Real data")
    plt.scatter(fake[:, 0], fake[:, 1], label="Fake data")

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
    cov_fake = np.eye(dimension)*0.1
    real_samples = 1000
    k_vals = [1,2,3]
    real_features = np.random.multivariate_normal(mean_real, cov_real, size=real_samples)
    fake_features = np.random.multivariate_normal(mean_real, cov_fake, size=real_samples)
    plotData(real_features, fake_features)
    score_dataframe = doEval(real_features, fake_features, k_vals)
    plotting.dataframeBoxplot(score_dataframe, "recall", "Original")
    plotting.dataframeBoxplot(score_dataframe, "precision", "Original")

    corner_value=4
    random_distribution = np.array([[-corner_value,-corner_value], [-corner_value, corner_value], [corner_value,-corner_value], [corner_value,corner_value]])
    fake_extra = np.concatenate((fake_features,random_distribution), axis=0)
    score_dataframe = doEval(real_features, fake_extra, k_vals)
    plotData(real_features, fake_extra)
    plotting.dataframeBoxplot(score_dataframe, "recall", "With corner points")
    plotting.dataframeBoxplot(score_dataframe, "precision", "With corner points")

doTest()
plt.show()