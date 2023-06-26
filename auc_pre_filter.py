import numpy as np
import pandas as pd
import check_densities as ch_den
import matplotlib.pyplot as plt
import sys

def filterByAUC(dataframe):
    dimensions = dataframe["dimension"].unique()
    for dim in dimensions:
        dimension_data = dataframe.loc[dataframe["dimension"] == dim, :]
        grouped = dimension_data.groupby(["dimensions_transformed"])["auc"].apply(
            lambda x: x.quantile(.75) - x.quantile(0.25))
        max_quantile_group = grouped.idxmax()
        filter_data = dimension_data.loc[dimension_data["dimensions_transformed"] == max_quantile_group, :]
        all_ids.extend(list(filter_data.index))

    filtered_dataframe = dataframe.loc[dataframe.index.isin(all_ids), :]
    filtered_dataframe.boxplot(column=['auc'], by=["dimension", "dimensions_transformed"])
    return filtered_dataframe

def savePairs(dataframe, dimensions, real_scaled, save_map):
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
    dataframe.to_pickle(f"{save_map}{sub_map}dataframe_{dimensions}.pkl")

all_ids = []
str_input = (sys.argv[1]).lower()
real_scaled = True
if str_input == "true":
    real_scaled = True
elif str_input == "false":
    real_scaled = False
save_path = "./dataframe_evaluation/"
if real_scaled:
    df_path = "./dataframe_factors/dataframe_real.pkl"
else:
    df_path = "./dataframe_factors/dataframe_fake.pkl"

dataframe = pd.read_pickle(df_path)
dataframe_filtered = filterByAUC(dataframe)
str_dimensions = sys.argv[2].split(',')
dimensions = [int(str_val) for str_val in str_dimensions]
print(real_scaled, dimensions)
savePairs(dataframe_filtered, dimensions, real_scaled, save_path)


