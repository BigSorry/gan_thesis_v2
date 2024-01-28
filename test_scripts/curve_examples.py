import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from utility_scripts import helper_functions as util
from create_data_scripts import check_densities as ch_den, create_eval_df as cr_eval_df
from experiments import experiment_calc as exp

import experiments.experiment_visualization as exp_vis

def doPlots(curve_var_dist, reference_distribution, scaled_distribution, lambda_factors,
            pr_pairs, dc_pairs, k_vals, save_path):
    real_z = -1
    fake_z = 1
    if lambda_factors[0] < lambda_factors[1]:
        real_z = 1
        fake_z = -1
    plt.subplot(1, 2, 1)
    exp_vis.plotTheoreticalCurve(curve_var_dist, curve_var_dist, lambda_factors, save=False)
    exp_vis.plotKNNMetrics(pr_pairs,  k_vals, "PR_KNN", "black", save_path, save=False)
    exp_vis.plotKNNMetrics(dc_pairs,  k_vals, "DC_KNN", "yellow", save_path, save=False)
    plt.legend(loc='upper center', bbox_to_anchor=(0, 1.15),
               fancybox=True, shadow=True, ncol=2, fontsize=9)
    plt.subplot(1, 2, 2)
    plt.subplots_adjust(wspace=.5)
    exp_vis.plotDistributions(reference_distribution, scaled_distribution, real_z, fake_z, "", save_path, save=True)


def getEvaluationPairs(scale_ratios_df, map_path):
    # Params setup
    scale_ratios = scale_ratios_df["ratio"].unique()
    scale_ratios = scale_ratios[scale_ratios < 0.9]
    scale_ratios = scale_ratios[scale_ratios > 0.1]
    real_scaling = True
    sample_size = 1000
    k_vals = [i for i in range(1, 8, 3)] + [99]
    print(k_vals)
    base_value = 1
    dimension = 2
    iters = 1
    scaling_mode = "real_scaled" if real_scaling else "fale_scaled"
    for iter in range(iters):
        for index, scale_factor in enumerate(scale_ratios):
            scale = base_value*scale_factor
            lambda_factors = [base_value, scale]
            reference_distribution, scaled_distribution = ch_den.getGaussian(sample_size, dimension, lambda_factors)
            if real_scaling:
                lambda_factors = [scale, base_value]
                distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = util.getDistanceMatrices(scaled_distribution, reference_distribution)
                curve_classifier = exp.getCurveClassifier("gaussian", scaled_distribution, reference_distribution, lambda_factors)
            else:
                distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = util.getDistanceMatrices(reference_distribution, scaled_distribution)
                curve_classifier = exp.getCurveClassifier("gaussian", reference_distribution, scaled_distribution, lambda_factors)

            auc = integrate.trapz(np.round(curve_classifier[:, 1], 2), np.round(curve_classifier[:, 0], 2))
            pr_pairs, dc_pairs = exp.getKNN(distance_matrix_real, distance_matrix_fake, distance_matrix_pairs, k_vals)
            # Real data -> scaled_distribution for examples
            save_path = f"{map_path}./explain_fig_scale_factor{scale_factor}.png"
            doPlots(curve_classifier, scaled_distribution, reference_distribution, lambda_factors,
                    pr_pairs, dc_pairs, k_vals, save_path)

def smallExperiment():
    real_scaled = True
    save_path_eval = "../curve_figures/"
    # Factors/ratios are pre-saved
    if real_scaled:
        scale_factors_path = "../dataframe_factors/dataframe_real.pkl"
    else:
        scale_factors_path = "../dataframe_factors/dataframe_fake.pkl"

    scale_factors_df = pd.read_pickle(scale_factors_path)
    scale_factors_df_sel = scale_factors_df.loc[scale_factors_df["dimension"] == 2, :]
    scale_factors_filtered = cr_eval_df.filterByAUC(scale_factors_df_sel)
    getEvaluationPairs(scale_factors_filtered, save_path_eval)

smallExperiment()
