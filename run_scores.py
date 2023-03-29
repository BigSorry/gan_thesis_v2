import numpy as np
import pandas as pd
from experiments import create_experiment as exp
from experiments import  distributions as dist
import matplotlib.pyplot as plt
import visualize as plotting
import experiments.experiment_visualization as exp_vis

def plotLine(dataframe, dimensions):
    grouped = dataframe.groupby(["dimension", "lambda_factor"]).mean().reset_index()
    score_names = ["precision", "recall", "density", "coverage"]
    score_names = ["recall", "coverage"]
    for dim in dimensions:
        sel_data = grouped.loc[grouped["dimension"] == dim, :]
        plt.figure()
        plt.title(f"Dimension {dim}")
        plt.xlabel("factor")
        plt.ylim([0, 1.1])
        for score_name in score_names:
            score_means = sel_data[score_name]
            factors = sel_data["lambda_factor"]
            plt.plot(factors, score_means, label=score_name)
        plt.legend()

    plt.show()


def getDistributions(iterations, sample_sizes, dimensions, lambda_factors, distribution_name):
    dict = {}
    for iter in range(iterations):
        for samples in sample_sizes:
            for dim in dimensions:
                key = (iter, samples, dim)
                reference_distribution, scaled_distributions = dist.getDensities(samples, dim, lambda_factors,
                                                                     distribution_name=distribution_name)
                dict[key] = {"reference":reference_distribution, "scaled": scaled_distributions}
    return dict
def gaussianLine():
    # Key is name which is scaled and value corresponding value for method calling
    scaling_info = {"fake_scaled": False, "real_scaled": True}
    scaling_info = {"fake_scaled": False}
    iterations = 2
    sample_sizes = [2000]
    sample_sizes = [1000]
    dimensions = [2, 8, 16, 32, 64]
    dimensions = [2, 8, 16, 32, 64]
    lambda_factors = np.array([1, 0.75, 0.5, 0.25, 0.1, 0.01])
    k_vals = [1]
    distribution_name = "gaussian"
    #map_path = f"C:/Users/lexme/Documents/gan_thesis_v2.2/gan_thesis_v2/{distribution_name}/{name}/"

    headers = ["iteration", "dimension", "lambda_factor", "k_val", "precision",
               "recall", "density", "coverage"]
    all_rows = []
    distribution_dict = getDistributions(iterations, sample_sizes, dimensions, lambda_factors, distribution_name)
    for key, value in distribution_dict.items():
        iter = key[0]
        dim = key[2]
        reference_distribution = value["reference"]
        scaled_distributions = value["scaled"]

        for index, scaled_distribution in enumerate(scaled_distributions):
            pr_pairs, dc_pairs = exp.getKNN(reference_distribution, scaled_distribution, k_vals)

            scale_factors = [lambda_factors[0], lambda_factors[index]]
            for i in range(len(k_vals)):
                precision = pr_pairs[i, 0]
                recall = pr_pairs[i, 1]
                density = dc_pairs[i, 0]
                coverage = dc_pairs[i, 1]


                all_rows.append([iter, dim, scale_factors[1], k_vals[i],
                                 precision, recall, density, coverage])

    dataframe = pd.DataFrame(data=all_rows, columns=headers)
    plotLine(dataframe, dimensions)

gaussianLine()