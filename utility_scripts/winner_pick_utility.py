import pandas as pd
import numpy as np

def filterGroupedData(group, best_mode):
    offset = 1e-3
    if best_mode:
        top_distance = group.loc[:, "distance"].min()
        boolean_filter = (group["distance"] <= top_distance + offset)
    else:
        top_distance = group.nlargest(1, "distance").loc[:, "distance"].max()
        boolean_filter = (group["distance"] >= top_distance) #| (
            #np.isclose(group["distance"], [top_distance], atol=1e-2))
    filter_data = group.loc[boolean_filter, :]

    return filter_data, top_distance

# Todo assume 999 k-value choices
def getBestValues(dataframe, max_winner, best_mode=True):
    k_values = []
    distance_values = []
    filter_data, top_distance = filterGroupedData(dataframe, best_mode)
    best_picks = filter_data["k_val"].values
    distances = filter_data["distance"].values
    # Short fix grouping same auc score within same iteration
    picked_rows = filter_data.shape[0]
    if picked_rows <= max_winner:
        for index, k_val in enumerate(best_picks):
            k_values.append(k_val)
            distance_values.append(distances[index])
    else:
        k_values.append(-1)
        distance_values.append(-1)

    return k_values, distance_values

def countTopPicks(dataframe, max_winner, best_mode):
    row_values = []
    grouped_data = dataframe.groupby(["iter", "dimension", "auc_score"])
    for (iter, dimension, auc_score), experiment_data in grouped_data:
        if experiment_data.shape[0] > 999:
            # Current fix, just take the first experiment (999 choices in one experiment)
            # print(f"Same auc score within the same experiment iteration id")
            # print(f"{iter}_{dimension}_{auc_score}")
            # print()
            experiment_data = experiment_data.iloc[:999, :]
        best_picks, distances = getBestValues(experiment_data, max_winner, best_mode=best_mode)
        count_value = 1 / len(best_picks)

        for index, k_value in enumerate(best_picks):
            distance = distances[index]
            values = [iter, auc_score, dimension, k_value, count_value, distance]
            row_values.append(values)

    dataframe_top_picks = pd.DataFrame(data=row_values, columns=["iter", "auc_score", "dimension",
                                                                 "k_value", "count_value", "distance"])

    return dataframe_top_picks


def countTopPicksDict(dataframe, filter_percentage, best_mode):
    dict_result = {}
    grouped_data = dataframe.groupby(["iter", "dimension", "auc_score"])
    for (iter, dimension, auc_score), group_data in grouped_data:
        best_picks, distances = getBestValues(group_data, filter_percentage, best_mode=best_mode)
        k_values = []
        distances = []
        for index, k_value in enumerate(best_picks):
            if k_value > -1:
                distance = distances[index]
                distances.append(distance)
                k_values.append(k_values)

        dict_result[(iter, dimension, auc_score)] = {}
        dict_result[(iter, dimension, auc_score)]["k_value"] = k_values
        dict_result[(iter, dimension, auc_score)]["distance"] = distances

    return dict_result

def getFilterMask(dataframe, max_winner=1):
    grouped_data = dataframe.groupby(["iter", "dimension", "auc_score"])
    excluded_keys = []
    for (iter, dimension, auc_score), experiment_data in grouped_data:
        if experiment_data.shape[0] > 999:
            experiment_data = experiment_data.iloc[:999, :]
        filter_data, top_distance = filterGroupedData(experiment_data, True)
        winner_rows = filter_data.shape[0]
        if winner_rows > max_winner:
            excluded_keys.append([iter, dimension, auc_score])

    filter_mask = np.ones(dataframe.shape[0]).astype(bool)
    for (iter, dimension, auc_score) in excluded_keys:
        excluded = (dataframe["iter"] == iter) & (dataframe["dimension"] == dimension) & (dataframe["auc_score"] == auc_score)
        filter_mask[excluded] = False

    return filter_mask