import numpy as np
import pandas as pd
import check_densities as ch_den
from scipy import integrate
from experiments import experiment_calc as exp

def checkAUCSpread(iters, dimensions, ratios_taken, real_scaling):
    base_value = 1
    sample_size = 1000
    ratios = np.round(np.linspace(.01, 1, ratios_taken), 4)
    rows = []
    for iter in range(iters):
        for dimension in dimensions:
            transform_dimensions = [int(dimension * 0.6), int(dimension * 0.7), int(dimension * 0.8), int(dimension * 0.9), dimension]
            uniques_transforms = np.unique(transform_dimensions)
            for transform_dimension in uniques_transforms:
                for index, ratio in enumerate(ratios):
                    scale = base_value * ratio
                    lambda_factors = [base_value, scale]
                    reference_distribution, scaled_distribution = ch_den.getGaussianDimension(sample_size, dimension, transform_dimension, lambda_factors)
                    if real_scaling:
                        lambda_factors = [scale, base_value]
                        curve_classifier = exp.getCurveClassifier("gaussian", scaled_distribution, reference_distribution, lambda_factors)
                    else:
                        curve_classifier = exp.getCurveClassifier("gaussian", reference_distribution, scaled_distribution, lambda_factors)
                    auc = integrate.trapz(np.round(curve_classifier[:, 1], 2), np.round(curve_classifier[:, 0], 2))
                    row = [iter, dimension, ratio, auc, transform_dimension]
                    rows.append(row)
    return rows

def saveDF():
    iters = 10
    ratios_taken = 50
    dimensions = [2**i for i in range(1, 10)]
    print(dimensions)
    real_scaled = True
    real_scaled_str = "real_scaled" if real_scaled else "fake_scaled"
    column_names = ["iter", "dimension", "ratio", "auc", "dimensions_transformed"]

    row_info = checkAUCSpread(iters, dimensions, ratios_taken, True)
    dataframe = pd.DataFrame(data=row_info, columns=column_names)
    df_path = "./pre_filter/dataframe_real.pkl"
    dataframe.to_pickle(df_path)
    print("real done")
    row_info = checkAUCSpread(iters, dimensions, ratios_taken, False)
    dataframe = pd.DataFrame(data=row_info, columns=column_names)
    df_path = "./pre_filter/dataframe_fake.pkl"
    dataframe.to_pickle(df_path)


