import os
import numpy as np
import pandas as pd
from experiments import experiment_calc as exp
from experiments import  distributions as dist
import matplotlib.pyplot as plt
import visualize as plotting
import experiments.experiment_visualization as exp_vis

def plotErrorbars(dataframe, dimensions, k_vals, score_names, map_path):
    do_extra_calc = False
    for score_name in score_names:
        score_map = f"{map_path}/{score_name}/"
        if not os.path.exists(score_map):
            os.makedirs(score_map)
        for dim in dimensions:
            save_path = f"{score_map}/dim{dim}.png"
            plt.figure()
            plt.title(f"Dimension {dim}")
            plt.xlabel("factor")
            sel_data = dataframe.loc[(dataframe["dimension"] == dim), :]
            grouped = sel_data.groupby(["k_val", "dimension", "lambda_factor"]).agg([np.mean, np.std]).reset_index()
            added_means = []
            for k_val in k_vals:
                sel_data = grouped[grouped["k_val"] == k_val]
                score_means = sel_data[score_name]["mean"]
                add_vector = True
                if k_val > 1 and do_extra_calc:
                    mean_values = np.array(added_means)
                    new_vector = np.reshape(score_means.values, (1, -1))
                    difference = np.sum(np.abs(mean_values - new_vector), axis=1)
                    check = np.isclose(difference, 0, atol=0.1).any()
                    if check:
                        add_vector=False
                if add_vector:
                    added_means.append(score_means.values)
                    score_std = sel_data[score_name]["std"]
                    factors = sel_data["lambda_factor"]
                    plt.errorbar(factors, score_means, score_std, linestyle='None', marker='o',
                                 label=f"{score_name}_k{k_val}")
            plt.legend()
            if score_name != "density":
                plt.ylim([0, 1.1])
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
def getDistributions(iterations, sample_sizes, dimensions, lambda_factors, distribution_name):
    dict = {}
    for iter in range(iterations):
        for samples in sample_sizes:
            for dim in dimensions:
                reference_distribution, scaled_distributions = dist.getDensities(samples, dim, lambda_factors,
                                                                     distribution_name=distribution_name)
                for index, scaled_distribution in enumerate(scaled_distributions):
                    key = (iter, samples, dim,lambda_factors[index])
                    dict[key] = {"reference":reference_distribution, "scaled": scaled_distribution}
    return dict
def getScores(target_distribution, scaled_distribution, k_vals, real_scaling=False):
    scores = []
    if real_scaling:
        pr_pairs, dc_pairs = exp.getKNN(scaled_distribution, target_distribution, k_vals)
    else:
        pr_pairs, dc_pairs = exp.getKNN(target_distribution, scaled_distribution, k_vals)
    for i in range(len(k_vals)):
        precision = pr_pairs[i, 0]
        recall = pr_pairs[i, 1]
        density = dc_pairs[i, 0]
        coverage = dc_pairs[i, 1]
        scores.append([k_vals[i], precision, recall, density, coverage])

    return scores
def makeRows(distribution_dict, k_vals, real_scaling=False):
    rows = []
    for key, distribution_info in distribution_dict.items():
        iter = key[0]
        dim = key[2]
        scaling_factor = key[3]
        reference_distribution = distribution_info["reference"]
        scaled_distribution = distribution_info["scaled"]
        # List of row scores where length depends on the amount of k-values
        scores = getScores(reference_distribution, scaled_distribution, k_vals, real_scaling)
        for score in scores:
            row = [iter, dim, scaling_factor] + score
            rows.append(row)

    return rows

def plotDistributions(distribution_dict):
    for key, distribution_info in distribution_dict.items():
        iter = key[0]
        dim = key[2]
        scaling_factor = key[3]
        unscaled = distribution_info["reference"]
        scaled = distribution_info["scaled"]
        if dim == 2:
            save_path = f"C:/Users/lexme/Documents/gan_thesis_v2.2/gan_thesis_v2/gaussian/fake_scaled/distribution/" \
                        f"scale{scaling_factor}.png"
            exp_vis.plotDistributions(unscaled, scaled, -1, 1, "", save_path, save=True)
            save_path = f"C:/Users/lexme/Documents/gan_thesis_v2.2/gan_thesis_v2/gaussian/real_scaled/distribution/" \
                        f"scale{scaling_factor}.png"
            exp_vis.plotDistributions(scaled, unscaled, 1, -1, "", save_path, save=True)


def getK(sample_size):
    small_boundary = 50
    max_boundary = 400
    k_small = [i for i in range(1, small_boundary, 1)]
    k_big = [i for i in range(small_boundary, max_boundary, 100)]
    k_vals = k_small + k_big

    return k_vals

def gaussianExperiment():
    # Key is name which is scaled and value corresponding value for method calling
    iterations = 2
    sample_sizes = [1000]
    dimensions = [2, 1024]
    lambda_factors = np.array([1, 0.75, 0.5, 0.25, 0.1, 0.01])
    lambda_factors = np.array([1, 0.1])
    k_vals = getK(sample_sizes[0])
    distribution_name = "gaussian"
    headers = ["iteration", "dimension", "lambda_factor", "k_val", "precision",
               "recall", "density", "coverage"]
    distribution_dict = getDistributions(iterations, sample_sizes, dimensions, lambda_factors, distribution_name)
    plotDistributions(distribution_dict)
    real_scaled = False

    fake_scaled_rows = makeRows(distribution_dict, k_vals, real_scaling=False)
    dataframe = pd.DataFrame(data=fake_scaled_rows, columns=headers)
    score_names = ["precision", "recall", "density", "coverage"]
    #score_names = ["recall"]
    map_path = "C:/Users/lexme/Documents/gan_thesis_v2.2/gan_thesis_v2/gaussian/fake_scaled/"
    exp_vis.plotLambaBoxplot(dataframe, dimensions, lambda_factors, score_names, map_path)
    #plotErrorbars(dataframe, dimensions, k_vals, score_names, map_path)

    if real_scaled:
        real_scaled_rows = makeRows(distribution_dict, k_vals, real_scaling=True)
        dataframe = pd.DataFrame(data=real_scaled_rows, columns=headers)
        score_names = ["precision", "recall", "density", "coverage"]
        score_names = ["precision", "recall", "density", "coverage"]
        map_path = "C:/Users/lexme/Documents/gan_thesis_v2.2/gan_thesis_v2/gaussian/real_scaled/"
        plotErrorbars(dataframe, dimensions, k_vals, score_names, map_path)


gaussianExperiment()
