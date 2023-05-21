import numpy as np
import helper_functions as util
import check_densities as ch_den
import gaussian_combine as gauss_combine


def runCombine(dimension, real_scaling):
    iters = 1
    sample_size = 1000
    k_vals = [i for i in range(1, sample_size, 10)]
    k_vals = [1, sample_size - 1]

    ratios = 2
    try_ratios = np.round(np.linspace(0.01, .99, ratios), 4)
    filter_std = 0.1
    if real_scaling:
        map_path = f"./gaussian_combine/paper_img/d{dimension}_real/"
    else:
        map_path = f"./gaussian_combine/paper_img/d{dimension}_fake/"

    ratios_path = f"./factors/d{dimension}_real_scaled_factors.pkl" if real_scaling \
        else f"./factors/d{dimension}_fake_scaled_factors.pkl"
    ratios = util.readPickle(ratios_path)
    calc_dict, _ = ch_den.doCalcs(sample_size, dimension, ratios, real_scaling=real_scaling)
    pr_row, dc_row = gauss_combine.combineResults(calc_dict)
    column_labels = ["distance_min", "distance_mean", "distance_max"]
    row_labels = ["All lambdas combined"]
    pr_colors = ch_den.getrowColors(pr_row)
    dc_colors = ch_den.getrowColors(dc_row)
    ch_den.plotTable(dimension, "pr", [pr_row], row_labels, column_labels, [pr_colors], map_path)
    ch_den.plotTable(dimension, "dc", [dc_row], row_labels, column_labels, [dc_colors], map_path)


def runGaussian(dimension, real_scaling):
    iters = 1
    sample_size = 1000
    k_vals = [i for i in range(1, sample_size, 10)]
    k_vals = [1, sample_size - 1]

    ratios = 10
    try_ratios = np.round(np.linspace(0.01, .99, ratios), 4)
    filter_std = 0.1
    if real_scaling:
        map_path = f"./gaussian_dimension/paper_img/d{dimension}_real/"
    else:
        map_path = f"./gaussian_dimension/paper_img/d{dimension}_fake/"

    # saveRatios(iters, k_vals, sample_size, dimension, try_ratios, filter_std, real_scaling=real_scaling)
    ratios_path = f"./factors/d{dimension}_real_scaled_factors.pkl" if real_scaling \
        else f"./factors/d{dimension}_fake_scaled_factors.pkl"
    ratios = util.readPickle(ratios_path)
    calc_dict, data_dict = ch_den.doCalcs(sample_size, dimension, ratios, real_scaling=real_scaling)
    # plotCurve(calc_dict, data_dict, dimension, map_path, real_scaling=real_scaling)
    ch_den. makeTable(dimension, calc_dict, map_path, real_scaling)


run_combine = True
dimension = 2
if run_combine:
    runCombine(dimension, real_scaling=True)
    runCombine(dimension, real_scaling=False)
    dimension = 64
    runCombine(dimension, real_scaling=True)
    runCombine(dimension, real_scaling=False)
else:
    runGaussian(dimension, real_scaling=True)
    runGaussian(dimension, real_scaling=False)
    dimension = 64
    runGaussian(dimension, real_scaling=True)
    runGaussian(dimension, real_scaling=False)
    # dimension = 1000
    # runExperiment(dimension, real_scaling=True)
    # runExperiment(dimension, real_scaling=False)