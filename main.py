import numpy as np
import helper_functions as util
import check_densities as ch_den
from pathlib import Path
from datetime import datetime
import sys

def makeCurves(dimensions, real_scaling):
    iters = 1
    sample_size = 1000
    k_vals = [i for i in range(1, sample_size, 1)]
    ratios_taken = 10
    ratios = np.round(np.linspace(.1, 1, ratios_taken), 4)

    for dimension in dimensions:
        if real_scaling:
            map_path = f"./gaussian_dimension/paper_img/curves/"
        else:
            map_path = f"./gaussian_dimension/paper_img/curves/"
        info_dict, data_dict = ch_den.getCurveData(iters, k_vals, sample_size, dimension, ratios,
                                                           real_scaling=real_scaling)
        ch_den.plotCurve(info_dict, map_path, real_scaling=real_scaling)

def runGaussian(dimensions, real_scaling):
    iters = 10
    sample_size = 1000
    k_vals = [i for i in range(1, sample_size, 1)]
    ratios_taken = 10
    ratios = np.round(np.linspace(0.1,  1, ratios_taken), 4)

    for dimension in dimensions:
        if real_scaling:
            map_path = f"./gaussian_dimension/paper_img/d{dimension}_real/"
            map_path = f"./gaussian_dimension/paper_img/tables_real/"
        else:
            map_path = f"./gaussian_dimension/paper_img/d{dimension}_fake/"
            map_path = f"./gaussian_dimension/paper_img/tables_fake/"
        pr_results, dc_results, data_dict = ch_den.doCalcs(iters, k_vals, sample_size, dimension, ratios, real_scaling=real_scaling)
        ch_den.makeTable("pr", dimension, pr_results, map_path, real_scaling)
        ch_den.makeTable("dc", dimension, dc_results, map_path, real_scaling)

def saveDict(score_dict, metric_name, experiment_config, auc_scores, map_path):
    saved_dict = {}
    k_distances = np.array(list(score_dict.values()))
    saved_dict["metric_name"] = metric_name
    saved_dict["distances"] = k_distances
    saved_dict["auc_scores"] = auc_scores
    saved_dict["experiment_config"] = experiment_config
    date_str = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = f"{map_path}d{experiment_config['dimension']}_{date_str}.pkl"
    util.savePickle(save_path, saved_dict)

def saveDistances(iters, dimensions, ratios_taken, pr_map_path, dc_map_path, real_scaling):
    sample_size = 1000
    ratios = np.round(np.linspace(.1, 1, ratios_taken), 4)
    k_vals = [i for i in range(1, sample_size, 1)]
    for dimension in dimensions:
        pr_k_results, dc_k_results, auc_scores = ch_den.getNearestDistances(iters, k_vals, sample_size,
                                                      dimension, ratios, real_scaling)
        auc_scores_np = np.array(auc_scores)
        experiment_config = {"iters":iters, "samples":sample_size, "ratios":ratios,
                             "real_scaled":real_scaled, "dimension":dimension}
        saveDict(pr_k_results, "pr", experiment_config, auc_scores_np, pr_map_path)
        saveDict(dc_k_results, "dc", experiment_config, auc_scores_np, dc_map_path)


iters = 2
ratios_taken = int(sys.argv[1])
dimensions_amount = list(sys.argv[2])
dimensions = [2**i for i in range(1, dimensions_amount+1)]
real_scaled = bool(int(sys.argv[3]))
real_scaled_str = "real_scaled" if real_scaled else "fake_scaled"
print(ratios_taken, dimensions)
print(real_scaled)

# pr_map_path = f"./factors/pr/{real_scaled_str}/"
# dc_map_path = f"./factors/dc/{real_scaled_str}/"
# Path(pr_map_path).mkdir(parents=True, exist_ok=True)
# Path(dc_map_path).mkdir(parents=True, exist_ok=True)
# saveDistances(iters, dimensions, ratios_taken,
#               pr_map_path, dc_map_path, real_scaled)
#
#
