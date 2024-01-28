import pandas as pd
from create_data_scripts import check_densities as ch_den
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# TODO quantile function should be absolute
def filterByAUC(dataframe):
    dimensions = dataframe["dimension"].unique()
    all_ids = []
    for dim in dimensions:
        dimension_data = dataframe.loc[dataframe["dimension"] == dim, :]
        grouped = dimension_data.groupby(["dimensions_transformed"])["auc"].apply(
            lambda x: np.abs(x.quantile(0.75) - x.quantile(0.25)))
        max_quantile_group = grouped.idxmax()
        filter_data = dimension_data.loc[dimension_data["dimensions_transformed"] == max_quantile_group, :]
        all_ids.extend(list(filter_data.index))

    filtered_dataframe = dataframe.loc[dataframe.index.isin(all_ids), :]
    filtered_dataframe.boxplot(column=['auc'], by=["dimension", "dimensions_transformed"])
    return filtered_dataframe

def saveEvaluationResults(dataframe, dimensions, real_scaled, save_map):
    column_names = ["metric_name", "scaling_mode", "iter", "dimension", "dimension_transformed",
                    "auc_score", "k_val", "distance", "first_score", "second_score"]
    iters = 10
    sample_size = 1000
    dimension_pre_filter = dataframe["dimension"].unique()
    dimensions_transformed_all = dataframe["dimensions_transformed"].unique()
    dim_to_dim_transformed = {dimension_pre_filter[i]: dimensions_transformed_all[i] for i in range(dimension_pre_filter.shape[0])}
    ratios = dataframe["ratio"].unique()
    k_vals = [i for i in range(1, sample_size, 1)]
    all_rows = []
    for index, dimension in enumerate(dimensions):
        dimensions_transformed = dim_to_dim_transformed[dimension]
        dimension_rows = ch_den.getEvaluationPairs(iters, k_vals, sample_size, dimension, dimensions_transformed, ratios, real_scaled)
        all_rows.extend(dimension_rows)
        print(dimension)

    dataframe = pd.DataFrame(data=all_rows, columns=column_names)
    sub_map = "real_scaled/" if real_scaled else "fake_scaled/"
    time = datetime.datetime.today().strftime('%Y-%m-%d')
    dataframe.to_pickle(f"{save_map}{sub_map}dataframe_{dimensions}_{time}.pkl")

def doFilter():
    str_input = (sys.argv[1]).lower()

    if str_input == "true":
        real_scaled = True
    elif str_input == "false":
        real_scaled = False
    save_path_eval = "../dataframe_evaluation/"

    # Factors/ratios are pre-saved
    # TODO seprate script/method for making factors
    if real_scaled:
         path_factors = "../dataframe_factors/dataframe_real.pkl"
    else:
        path_factors = "../dataframe_factors/dataframe_fake.pkl"

    dataframe = pd.read_pickle(path_factors)
    dataframe_filtered = filterByAUC(dataframe)
    str_dimensions = sys.argv[2].split(',')
    dimensions = [int(str_val) for str_val in str_dimensions]
    print(real_scaled, dimensions)
    saveEvaluationResults(dataframe_filtered, dimensions, real_scaled, save_path_eval)
def showRatios(dataframe):
    all_ratios = []
    for value in dataframe["ratio"].values:
        print(value)
def checkAUCSpread(dataframe, scaling_mode):
    dimensions = dataframe["dimension"].unique()
    map_path = f"auc_spread/{scaling_mode}/"
    Path(map_path).mkdir(parents=True, exist_ok=True)
    for dim in dimensions:
        plt.title(f"Dimension {dim}")
        dimension_data = dataframe.loc[dataframe["dimension"] == dim, :]
        dimension_data.boxplot(column=['auc'], by=["dimensions_transformed"])
        save_path = f"{map_path}dimension{dim}.png"
        plt.savefig(save_path)
        plt.close()
def checkLambdaScaling(dataframe, scaling_mode):
    dimensions = dataframe["dimension"].unique()
    iterations = dataframe["iter"].unique()
    map_path = f"lambda_scalings/{scaling_mode}/"
    Path(map_path).mkdir(parents=True, exist_ok=True)
    values = []
    for dim in dimensions:
        for iter in iterations:
            sel_data = dataframe.loc[(dataframe["dimension"] == dim) & (dataframe["iter"] == iter), :]
            lambdas = sel_data["ratio"]
            for val in lambdas:
                if val not in values:
                    values.append(val)
            if lambdas.shape[0] != 50:
                print(lambdas.shape)
    print(values)
def getSavedFactors():
    path_real_factors = "../dataframe_factors/dataframe_real.pkl"
    path_fake_factors = "../dataframe_factors/dataframe_fake.pkl"
    paths = {"real_scaled": path_real_factors, "fake_scaled": path_fake_factors}

    for scaling_mode, path in paths.items():
        dataframe = pd.read_pickle(path)
        dataframe_filtered = filterByAUC(dataframe)
        checkAUCSpread(dataframe_filtered, scaling_mode)
        # checkLambdaScaling(dataframe_filtered, scaling_mode)
        # print(dataframe_filtered["dimensions_transformed"].unique())


getSavedFactors()

# By variance
# [  2   4   8  16  32  64 115 204 409]
# [  2   4   8  16  32  64 115 153 307]

# By IQR
# [  2   4   8  16  32  57 102 179 358]
# [  2   4   8  16  32  57 102 153 307]

# By IQR absolute
# [  2   4   8  16  32  57 102 179 358]
# [  2   4   8  16  32  57 102 153 307]