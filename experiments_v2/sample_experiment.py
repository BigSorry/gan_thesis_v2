import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visualize as plotting
import experiments_v2.helper_functions as util

def getKParams(sample_size, max_k, step_size=10):
    vals = []
    for i in range(1, max_k, step_size):
        size = i
        vals.append(size)
    if vals[len(vals) - 1] < (sample_size - 1):
        vals.append(sample_size-1)
    return vals

def getData(iters, dimensions, sample_sizes, lambda_factors):
    columns = ["iter", "sample_size", "dimension", "lambda",
               "k_val", "recall", "coverage"]
    row_data = []
    for iter in range(iters):
        for samples in sample_sizes:
            k_vals = getKParams(samples, max_k=int(samples), step_size=20)
            print(k_vals)
            for dimension in dimensions:
                mean_vec = np.zeros(dimension)
                for scale_factor in lambda_factors:
                    cov_real = np.eye(dimension)
                    cov_fake = np.eye(dimension) * scale_factor
                    real_features = np.random.multivariate_normal(mean_vec, cov_real, size=samples)
                    fake_features = np.random.multivariate_normal(mean_vec, cov_fake, size=samples)
                    distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = util.getDistanceMatrices(
                        real_features, fake_features)
                    for k_val in k_vals:
                        # Calculations
                        boundaries_real = distance_matrix_real[:, k_val]
                        boundaries_fake = distance_matrix_real[:, k_val]
                        precision, recall, density, coverage = util.getScores(distance_matrix_pairs, boundaries_fake,
                                                                              boundaries_real, k_val)
                        row = [iter, samples, dimension, scale_factor, k_val, recall, coverage]
                        row_data.append(row)

    dataframe = pd.DataFrame(columns=columns, data=row_data)
    return dataframe

def plotExperiment(dataframe, sample_size, dimension, show_boxplot,
                   show_map, path_box, path_map):
    effective_sample_size = sample_size / dimension
    recall_text = f"Recall with sample size {sample_size}, dimension {dimension}, and ess {effective_sample_size}"
    coverage_text = f"Recall with sample size {sample_size}, dimension {dimension}, and ess {effective_sample_size}"
    recall_file = f"recall_samples{sample_size}_dim{dimension}.png"
    coverage_file = f"coverage_samples{sample_size}_dim{dimension}.png"
    if show_boxplot:
        grouped_data = dataframe.groupby(["k_val"])
        recalls = []
        coverages = []
        xticks = []
        for k_val, group_data in grouped_data:
            recall = group_data["recall"].values
            coverage = group_data["coverage"].values
            recalls.append(recall)
            coverages.append(coverage)
            xticks.append(k_val)

        plotting.plotBox(recalls, xticks, recall_text,
                         save=True, save_path=path_box+recall_file)
        plotting.plotBox(coverages, xticks, coverage_text,
                         save=True, save_path=path_box+coverage_file)

    if show_map:
        recall_pivot = pd.pivot_table(dataframe, values='recall', index=['lambda'],
                columns=['k_val'], aggfunc=np.mean)
        coverage_pivot = pd.pivot_table(dataframe, values='coverage', index=['lambda'],
                                      columns=['k_val'], aggfunc=np.mean)

        plotting.HeatMapPivot(recall_pivot, title_text=recall_text,
                              save=True, save_path=path_map+recall_file)
        plotting.HeatMapPivot(coverage_pivot, title_text=coverage_text,
                              save=True, save_path=path_map+coverage_file)


import seaborn as sns
def sampleExperiment():
    # Data Setup
    iters = 2
    dimensions = [2, 4, 8, 16, 32, 64]
    sample_sizes = [1000, 2000, 4000, 8000]
    lambda_factors = [0.01, 0.1, 10, 100]
    save_data = True
    path_data = "C:/Users/Lex/OneDrive/dataframe/dataframe.pickle"
    if save_data:
        dataframe = getData(iters, dimensions, sample_sizes, lambda_factors)
        util.savePickle(path_data, dataframe)
    else:
        dataframe = util.readPickle(path_data)

    show_box=True
    show_map=True
    # PC paths
    path_map = "C:/Users/Lex/OneDrive/plots_thesis/pc/heatmap/"
    path_box = "C:/Users/Lex/OneDrive/plots_thesis/pc/boxplot/"
    # path_map = "C:/Users/lexme/OneDrive/plots_thesis/laptop/heatmap/"
    # path_box = "C:/Users/lexme/OneDrive/plots_thesis/laptop/boxplot/"
    for sample_size in sample_sizes:
        for dimension in dimensions:
                select_data = dataframe.loc[(dataframe["sample_size"] == sample_size) &
                                            (dataframe["dimension"] == dimension) , :]
                plotExperiment(select_data, sample_size, dimension, show_box,
                               show_map, path_box, path_map)


def checkData():
    path_data = "C:/Users/Lex/OneDrive/dataframe/dataframe.pickle"
    dataframe = util.readPickle(path_data)
    sample_sizes = dataframe["sample_size"].unique()
    dimensions = dataframe["dimension"].unique()
    median_recalls = np.zeros((sample_sizes.shape[0], dimensions.shape[0]))
    median_coverages = np.zeros((sample_sizes.shape[0], dimensions.shape[0]))
    for sample_id, sample_size in enumerate(sample_sizes):
        for dimension_id, dimension in enumerate(dimensions):
            select_data = dataframe.loc[(dataframe["sample_size"] == sample_size) &
                                        (dataframe["dimension"] == dimension), :]
            median_recall = select_data["recall"].std()
            median_coverage = select_data["coverage"].std()
            median_recalls[sample_id, dimension_id] = median_recall
            median_coverages[sample_id, dimension_id] = median_coverage

    plotting.saveHeatMap(median_recalls, dimensions, sample_sizes, title_text="Recall")
    plotting.saveHeatMap(median_coverages, dimensions, sample_sizes, title_text="Coverage")

def checkDistancesData():
    iters = 2
    dimensions = [2]
    sample_sizes = [1000]
    lambda_factors = [0.01, 1000]
    k_vals = [1 , 999]
    for samples in sample_sizes:
        #k_vals = getKParams(samples, max_k=int(samples), step_size=20)
        for dimension in dimensions:
            mean_vec = np.zeros(dimension)
            for scale_factor in lambda_factors:
                index = 0
                for k in k_vals:
                    cov_real = np.eye(dimension)
                    cov_fake = np.eye(dimension) * scale_factor
                    real_features = np.random.multivariate_normal(mean_vec, cov_real, size=samples)
                    fake_features = np.random.multivariate_normal(mean_vec, cov_fake, size=samples)
                    distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = util.getDistanceMatrices(
                        real_features, fake_features)
                    boundaries_real = distance_matrix_real[:, k]
                    boundaries_fake = distance_matrix_fake[:, k]
                    columns_sorted = np.sort(distance_matrix_pairs, axis=0)
                    rows_sorted = np.sort(distance_matrix_pairs, axis=1)
                    fake_between = columns_sorted[k, :]


                    real_between = rows_sorted[:, k]
                    precision, recall, density, coverage = util.getScores(distance_matrix_pairs, boundaries_fake,
                                                                          boundaries_real, k)


                    print(recall, coverage)
                    fig, ax = plt.subplots(2,2, sharey=True)
                    fig.suptitle(f"Samples is {samples} and dimension is {dimension}, "
                                 f"lambda is {scale_factor} and K_val is {k}"
                                 f"\n \n Coverage is {coverage:.2f} and Recall is {recall:.2f}")
                    ax = ax.flatten()
                    ax[0].boxplot(boundaries_real.flatten(), positions=[index])
                    ax[0].set_title("Boundaries real distances")
                    ax[1].boxplot(fake_between.flatten(), positions=[index])
                    ax[1].set_title("kth real neighbour for fake samples")
                    ax[2].set_title("Boundaries fake distances")
                    ax[2].boxplot(boundaries_fake.flatten(), positions=[index])
                    ax[3].set_title("kth fake neighbour for real samples")
                    ax[3].boxplot(real_between.flatten(), positions=[index])
                    index+=1

                plt.xticks(np.arange(len(lambda_factors))+1, lambda_factors)

def explainProblem():
    iters = 2
    dimensions = [2]
    sample_sizes = [1000]
    lambda_factors = [0.1]
    k_vals = [1]
    show_box = True
    show_points = True
    save_path = "./fig_v2/explain_methods/"
    for samples in sample_sizes:
        for dimension in dimensions:
            mean_vec = np.zeros(dimension)
            for scale_factor in lambda_factors:
                for k in k_vals:
                    mean_real = mean_vec
                    cov_real = np.eye(dimension)
                    cov_fake = np.eye(dimension) * scale_factor
                    real_features = np.random.multivariate_normal(mean_real, cov_real, size=samples)
                    fake_features = np.random.multivariate_normal(mean_vec, cov_fake, size=samples)
                    distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = util.getDistanceMatrices(
                        real_features, fake_features)
                    boundaries_real = distance_matrix_real[:, k]
                    boundaries_fake = distance_matrix_fake[:, k]
                    fakes_real_neighbour = distance_matrix_pairs.min(axis=0)
                    reals_fake_neighbour = distance_matrix_pairs.min(axis=1)
                    boolean_filter = filterRecall(distance_matrix_pairs, boundaries_fake)
                    boolean_filter = ~filterCoverage(distance_matrix_pairs, boundaries_real)

                    precision, recall, density, coverage = util.getScores(distance_matrix_pairs[boolean_filter, :], boundaries_fake,
                                                                          boundaries_real[boolean_filter], k)
                    recall_mask, coverage_mask = util.getScoreMask(boundaries_real[boolean_filter], boundaries_fake,
                                                                     distance_matrix_pairs[boolean_filter, :])

                    print(f"Recall is {recall} and Coverage is {coverage}")
                    title_text = f"Samples is {samples}, dimension is {dimension}, "\
                                 f"lambda is {scale_factor}, and k_val is {k}\n"\
                                 f"Recall is {recall:.2f} and Coverage is {coverage:.2f} \n\n"

                    print(f"Filtered {boolean_filter.mean()}")
                    if show_box:
                        plotBox(boundaries_real[boolean_filter], boundaries_fake, reals_fake_neighbour[boolean_filter], title_text,
                                ["Real samples kth \n real neighbour distance",
                                 "Fake samples kth \n fake neighbour distance",
                                 "Real samples smallest \n distance to fake samples"],
                                True, save_path)
                    if show_points:
                        plotting.plotData(real_features[boolean_filter, :], fake_features,
                                          boundaries_real[boolean_filter], boundaries_fake,
                                 recall_mask, coverage_mask, title_text, True, save_path)

def filterRecall(distance_matrix, boundaries):
    boolean_matrix = distance_matrix > boundaries
    boolean_filter = boolean_matrix.all(axis=1)

    return boolean_filter

def filterCoverage(distance_matrix, boundaries):
    boolean_matrix = distance_matrix < np.expand_dims(boundaries, axis=1)
    boolean_filter = boolean_matrix.any(axis=1)

    return boolean_filter

def plotBox(real_boundaries, fake_boundaries, other_boundaries,
            title_text, xticks_labels, save=False, save_path=""):
    plt.figure(figsize=(8, 5))
    plt.title(title_text)
    plt.boxplot([real_boundaries, fake_boundaries, other_boundaries])
    plt.axhline(y=np.max(fake_boundaries), color='r', linestyle='-')
    plt.xticks([1, 2, 3], xticks_labels)

    if save:
        plt.subplots_adjust(wspace=0.3)
        plt.savefig(f"{save_path}boxplot.png",
                    dpi=300, bbox_inches='tight')
        plt.close()