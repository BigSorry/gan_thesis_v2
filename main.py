import numpy as np
import helper_functions as util
import check_densities as ch_den
import gaussian_combine as gauss_combine


# filter_std = 0.1
# ch_den.saveRatios(iters, k_vals, sample_size, dimension, try_ratios, filter_std, real_scaling=real_scaling)
# ratios_path = f"./factors/d{dimension}_real_scaled_factors.pkl" if real_scaling \
#     else f"./factors/d{dimension}_fake_scaled_factors.pkl"
# ratios = util.readPickle(ratios_path)

def runCombine(dimension, real_scaling):
    iters = 1
    sample_size = 1000
    if real_scaling:
        map_path = f"./gaussian_combine/paper_img/d{dimension}_real/"
    else:
        map_path = f"./gaussian_combine/paper_img/d{dimension}_fake/"

    take_ratios = 5
    ratios = np.round(np.linspace(0.01, .99, take_ratios), 4)
    ratios_path = f"./factors/d{dimension}_real_scaled_factors.pkl" if real_scaling \
        else f"./factors/d{dimension}_fake_scaled_factors.pkl"
    #ratios = util.readPickle(ratios_path)
    pr_results, dc_results, data_dict = gauss_combine.doCalcs(sample_size, dimension, ratios, real_scaling=real_scaling)
    gauss_combine.saveBoxplot(pr_results, "pr", map_path)
    gauss_combine.saveBoxplot(dc_results, "dc", map_path)


def runGaussian(dimensions, real_scaling):
    iters = 10
    sample_size = 1000
    k_vals = [i for i in range(1, 100, 10)]
    k_vals = [1, sample_size - 1]
    ratios_taken = 20
    ratios = np.round(np.linspace(0.1,  1, ratios_taken), 4)

    # ch_den.plotCurve(calc_dict, data_dict, dimension, map_path, real_scaling=real_scaling)
    for dimension in dimensions:
        if real_scaling:
            map_path = f"./gaussian_dimension/paper_img/d{dimension}_real/"
        else:
            map_path = f"./gaussian_dimension/paper_img/d{dimension}_fake/"
        pr_results, dc_results, data_dict = ch_den.doCalcs(iters, k_vals, sample_size, dimension, ratios, real_scaling=real_scaling)
        ch_den.makeTable("pr", dimension, pr_results, map_path, real_scaling)
        ch_den.makeTable("dc", dimension, dc_results, map_path, real_scaling)

def gausianBestK(dimensions, real_scaling):
    iters = 5
    sample_size = 1000
    ratios_taken = 10
    ratios = np.round(np.linspace(0.1, 1, ratios_taken), 4)
    k_vals = [i for i in range(1, 100, 5)]
    k_vals = [1, 3, 5, 7, 10, 50, sample_size // 4, sample_size // 2 ,sample_size - 1]
    if real_scaling:
        map_path = f"./gaussian_dimension/paper_img/boxplots/d64_real/"
    else:
        map_path = f"./gaussian_dimension/paper_img/boxplots/d64_fake/"

    # try_separate = True
    # if try_separate:
    #     for ratio in ratios:
    #         pr_calc, dc_calc = ch_den.getNearestDistances(iters, k_vals, sample_size,
    #                                                       dimension, [ratio], real_scaling)
    #         k_vals_np = np.array(k_vals)
    #         ch_den.saveBoxplot(pr_calc, k_vals_np, f"pr_{ratio}", map_path)
    #         ch_den.saveBoxplot(dc_calc, k_vals_np, f"dc__{ratio}", map_path)

    for dimension in dimensions:
        pr_k_results, dc_k_results = ch_den.getNearestDistances(iters, k_vals, sample_size,
                                                      dimension, ratios, real_scaling)
        k_vals_np = np.array(k_vals)
        ch_den.saveBoxplot(pr_k_results, k_vals_np, f"pr_all_d{dimension}", map_path)
        ch_den.saveBoxplot(dc_k_results, k_vals_np, f"dc__all_d{dimension}", map_path)

def runExperiments():
    run_combine = False
    dimensions = [2, 64, 1000]
    dimensions = [64]
    for dimension in dimensions:
        if run_combine:
            runCombine(dimension, real_scaling=True)
            runCombine(dimension, real_scaling=False)
        else:
            runGaussian(dimension, real_scaling=True)
            runGaussian(dimension, real_scaling=False)

dimensions = [2]
runGaussian(dimensions, True)
# gausianBestK(dimensions, True)
# gausianBestK(dimensions, False)

