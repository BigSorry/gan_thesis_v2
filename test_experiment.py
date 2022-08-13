import datetime
import metrics.metrics as mtr
import numpy as np
import matplotlib.pyplot as plt
import experiment_utils as util
import visualize as plotting
from CreatorExperiment import CreateExperiment
from sklearn.metrics import pairwise_distances

def getBoundaries(real_features, fake_features, k):
    distance_matrix_real = pairwise_distances(real_features, real_features, metric='euclidean')
    distance_matrix_fake = pairwise_distances(fake_features, fake_features, metric='euclidean')
    distance_matrix_pairs = pairwise_distances(real_features, fake_features, metric='euclidean')
    distance_matrix_real = np.sort(distance_matrix_real, axis=1)
    distance_matrix_fake = np.sort(distance_matrix_fake, axis=1)
    boundaries_real = distance_matrix_real[:, k]
    boundaries_fake = distance_matrix_fake[:, k]

    return boundaries_real, boundaries_fake, distance_matrix_pairs

def showBoundaryHistogram(boundaries_real, boundaries_fake, k, bins=20):
    x_max = max(np.max(boundaries_fake), np.max(boundaries_real)) + 0.1
    plt.suptitle(f"K is {k}")
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_title("Real data radii histogram")
    ax1.hist(boundaries_real, bins=bins, density=True, color='r')
    ax1.set_xlabel("Distance")
    ax1.set_ylabel("Density")
    ax1.set_xlim([0, x_max])

    ax2 = plt.subplot(2, 2, 2)
    ax2.set_title("Fake data radii histogram")
    ax2.hist(boundaries_fake, bins=bins, density=True, color='g')
    ax2.set_xlabel("Distance")
    ax2.set_ylabel("Density")
    ax2.set_xlim([0, x_max])

def plotScores(pr_score, pr_score2):
    plt.subplot(2, 2, 3)
    width = 0.3
    x_names = ["Precision Recall", "Density Coverage"]
    x_axes = np.arange(len(x_names)) + 1
    scores = [pr_score, pr_score2]
    for i, score in enumerate(scores):
        position = x_axes[i]
        pr_score = scores[i]
        plt.bar(position - (width/2), pr_score[0], width, color='b')
        plt.bar(position + (width/2), pr_score[1], width, color='g')

    plt.xticks(x_axes, x_names)

def plotData(real_features, fake_features):
    plt.scatter(real_features[:, 0], real_features[:, 1], label="real data", c="r")
    plt.scatter(fake_features[:, 0], fake_features[:, 1], label="fake data", c="g")
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.legend()

def plotCircles(points, boundaries):
    plt.figure()
    normalized = (boundaries - np.min(boundaries)) / (np.max(boundaries) - np.min(boundaries))
    for i in range(len(points)):
        if i % 100 == 0:
            point = points[i]
            radii = normalized[i]
            plt.scatter(point[0], point[1], s=radii, facecolors='none', edgecolors='r')


def gaussianExperiment(samples, scales_fake, k_val, visualize=True):
    real_features = np.random.normal(loc=0, scale=1, size=[samples, 2])
    fake_features = np.random.normal(loc=0, scale=scales_fake, size=[samples, 2])
    used_bins = 40
    boundaries_real, boundaries_fake, distance_matrix_pairs = getBoundaries(real_features, fake_features, k_val)
    # knn
    precision = (distance_matrix_pairs < np.expand_dims(boundaries_real, axis=1)).any(axis=0).mean()
    recall = (distance_matrix_pairs < np.expand_dims(boundaries_fake, axis=0)).any(axis=1).mean()
    # density coverage
    density = (1. / float(k_val)) * (distance_matrix_pairs < np.expand_dims(boundaries_real, axis=1)).sum(axis=0).mean()
    coverage = (distance_matrix_pairs.min(axis=1) < boundaries_real).mean()
    pr_score1 = np.array([precision, recall])
    pr_score2 = np.array([density, coverage])

    if visualize:
        #plotCircles(fake_features, boundaries_fake)
        fig = plt.figure()
        plotData(real_features, fake_features)
        showBoundaryHistogram(boundaries_real, boundaries_fake, k_val, bins=used_bins)
        plotScores(pr_score1, pr_score2)
        fig.tight_layout()
        fig.savefig(f"./fig_v2/data/k{k_val}_scale{scales_fake}.png", dpi=250)

    return pr_score1, pr_score2


def plotScaleData(data_dict):
    for k_param, scale_dict in data_dict.items():
        fig = plt.figure()
        fig.suptitle(f"K is {k_param}")
        for scale_param, score_dict in scale_dict.items():
            index = 1
            for metric_name, scores in score_dict.items():
                plt.subplot(2, 2, index)
                mean_score = np.mean(scores)
                std = np.std(scores)
                plt.errorbar(scale_param, mean_score, std, color='blue', fmt='o')
                index += 1

        ax1 = plt.subplot(2, 2, 1)
        ax1.set_title("Precision")
        ax1.set_ylim([0, 1.1])
        ax1.set_xlabel('Variance param fake Gaussian')
        ax2 = plt.subplot(2, 2, 2)
        ax2.set_ylim([0, 1.1])
        ax2.set_title("Recall")
        ax2.set_xlabel('Variance param fake Gaussian')
        ax3 = plt.subplot(2, 2, 3)
        ax3.set_ylim([0, 1.1])
        ax3.set_title("Density")
        ax3.set_xlabel('Variance param fake Gaussian')
        ax4 = plt.subplot(2, 2, 4)
        ax4.set_ylim([0, 1.1])
        ax4.set_title("Coverage")
        ax4.set_xlabel('Variance param fake Gaussian')

        fig.tight_layout()
        fig.savefig(f"./fig_v2/k{k_param}.png", dpi=250)

def doExperiments():
    samples = 1000
    scales_fake = [0.1, 0.25, 0.5, 1, 2, 4, 8, 100]
    scales_fake = [0.1,  8, 100]
    k_vals = [1,  8, 16]

    #scales_fake = [0.1, 1, 8]
    #k_vals = [8]
    data_dict = {}
    iters = 1
    for i in range(iters):
        for k_val in k_vals:
            if k_val not in data_dict:
                data_dict[k_val] = {}
            for scale in scales_fake:
                pr_score1, pr_score2 = gaussianExperiment(samples, scale, k_val, visualize=True)

                if scale not in data_dict[k_val]:
                    data_dict[k_val][scale] = {}
                    data_dict[k_val][scale]["precision"] = []
                    data_dict[k_val][scale]["recall"] = []
                    data_dict[k_val][scale]["density"] = []
                    data_dict[k_val][scale]["coverage"] = []

                data_dict[k_val][scale]["precision"].append(pr_score1[0])
                data_dict[k_val][scale]["recall"].append(pr_score1[1])
                data_dict[k_val][scale]["density"].append(pr_score2[0])
                data_dict[k_val][scale]["coverage"].append(pr_score2[1])

    plotScaleData(data_dict)



# doExperiments()

plt.show()