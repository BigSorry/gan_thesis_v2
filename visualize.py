import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import experiments_v2.helper_functions as helper
import seaborn as sns

def plotScores(result_dict):
    plt.title("Recall vs Coverage")
    width = 0.2
    x_names = list(result_dict.keys())
    x_axes = np.arange(len(x_names))*0.5
    index = 0
    for method_name, method_score, in result_dict.items():
        position = x_axes[index]
        plt.bar(position, method_score, width, color='g')
        plt.annotate(f"{method_score:0.2f}", (position-0.01, method_score+0.01),
                     fontsize=14, color="black")
        index+=1

    plt.ylim([0, 1.1])
    plt.xticks(x_axes, x_names)

def plotCircles(data, boundaries):
    alpha_val = 0.2
    for index, sample in enumerate(data):
        radius = boundaries[index]
        #fill_circle = plt.Circle((sample[0], sample[1]), radius, color='yellow', fill=True, alpha=alpha_val)
        circle_boundary = plt.Circle((sample[0], sample[1]), radius, color='black', fill=False)
        #plt.gca().add_patch(fill_circle)
        plt.gca().add_patch(circle_boundary)

def plotAcceptRejectData(data, boolean_mask, data_kind="real"):
    alpha_val = 1
    accepted_data = data[boolean_mask, :]
    rejected_data = data[~boolean_mask, :]
    plt.scatter(accepted_data[:, 0], accepted_data[:, 1], c="green", label=f"Accepted {data_kind} data", zorder=98, s=2**4, alpha=alpha_val)
    plt.scatter(rejected_data[:, 0], rejected_data[:, 1], c="red", label=f"Rejected {data_kind} data", zorder=97, s=2**4, alpha=alpha_val)


def _plotErrorbar(subplot, title_text, x, scores):
    score_mean = np.mean(scores)
    score_std = np.std(scores)
    subplot.set_title(title_text)
    # TODO X label changes for some experiments
    subplot.set_xlabel('K values')
    subplot.set_ylabel('Score')
    subplot.set_ylim([0, 1.1])
    subplot.errorbar(x, score_mean, score_std, linestyle='None', marker='o', color='blue')

def showScores(result_dict, save=False, save_path="", subplots=2):
    fig, ax = plt.subplots(subplots)
    for radi, scores in result_dict.items():
        scores_np = np.array(scores)
        recalls = scores_np[:, 1]
        _plotErrorbar(ax[0], "recall", radi, recalls)
        coverages = scores_np[:, 3]
        _plotErrorbar(ax[1], "coverage", radi, coverages)

        if subplots == 4:
            precisions = scores_np[:, 0]
            _plotErrorbar(ax[2], "precision", radi, precisions)
            densities = scores_np[:, 2]
            _plotErrorbar(ax[3], "density", radi, densities)
    if save:
        plt.tight_layout()
        fig.savefig(f"{save_path}all_results.png", dpi=600)

def getAnnotColors(annotations):
    color_map = []
    for i in range(annotations.shape[0]):
        for j in range(annotations.shape[1]):
            pass

def HeatMapPivot(pivot_table, title_text="", save=False, save_path=""):
    plt.title(title_text)
    ax = sns.heatmap(pivot_table, cmap="RdYlGn_r", annot=True,
                annot_kws={"color": "black", "backgroundcolor": "white"},
                     vmin=0, vmax=1)
    plt.yticks(rotation=0)
    ax.invert_yaxis()
    if save:
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.savefig(save_path, dpi=300)
        plt.close()

def saveHeatMap(tabular_data, rows, columns, save=False, save_path="", title_text=""):
    plt.figure(figsize=(14, 6))
    sns.heatmap(tabular_data,
                cmap="RdYlGn_r",
                xticklabels=rows,
                yticklabels=columns,
                annot=tabular_data,
                annot_kws={"color":"black", "backgroundcolor":"white"},
                vmin=0,
                vmax=1
                )
    plt.title(title_text)
    plt.yticks(rotation=0)

    if save:
        plt.subplots_adjust(wspace=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def plotDistanceHisto(real, fake, title):
    plt.figure()
    plt.suptitle(title)
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title("Real boundaries")
    plt.hist(real, density=True)

    ax2 = plt.subplot(1, 2, 2, sharex=ax1)
    ax2.set_title("Fake boundaries")
    plt.hist(fake, density=True)

def plotDataFrame(dataframe, metric_name):
    grouped = dataframe.groupby("dimension")
    for name, group in grouped:
        variances = group["variance"].unique()
        k_vals = group["k_val"].unique()
        rows =[f"K={k_val}" for k_val in k_vals]
        columns = [f"\u03BB={var}" for var in variances]
        cell_data = group.pivot(index="variance", columns="k_val", values=metric_name)
        title_text = f"{metric_name}, dimension is {name}"
        saveHeatMap(cell_data.values, rows, columns, save=True, save_path=f"./fig_v2/mode_drop/{title_text}.png", title_text=title_text)

# input pandas dataframe
def plotScoreErrorbars(score_names, mean_data, std_data, title_text):
   plt.figure()
   plt.title(title_text)
   dimensions = mean_data.index
   for score_name in score_names:
       score_mean = mean_data[score_name]
       score_std = std_data[score_name]
       plt.errorbar(dimensions, score_mean, score_std, label=score_name)

   plt.ylim([0, 1.1])
   plt.xlabel("dimension")
   plt.legend()

def plotDistances(mean_data, std_data):
   plt.figure()
   dimensions = mean_data.index
   avg_difference_real = mean_data["avg_difference_real"]
   avg_difference_std = std_data["avg_difference_real"]
   avg_difference_fake = mean_data["avg_difference_fake"]
   avg_difference_fake_std = std_data["avg_difference_fake"]
   plt.errorbar(dimensions, avg_difference_real, avg_difference_std, label="Mean absolute difference min-max distance real")
   plt.errorbar(dimensions, avg_difference_fake, avg_difference_fake_std, label="Mean absolute difference min-max distance fake")
   plt.legend()

def setLimits(real_data, fake_data, boundaries_real, boundaries_fake):
    min_x = min(np.min(real_data[:, 0]), np.min(fake_data[:, 0]))
    max_x = max(np.max(real_data[:, 0]), np.max(fake_data[:, 0]))
    min_y = min(np.min(real_data[:, 1]), np.min(fake_data[:, 1]))
    max_y = max(np.max(real_data[:, 1]), np.max(fake_data[:, 1]))
    max_boundary = max(np.max(boundaries_real), np.max(boundaries_fake))
    offset = max_boundary
    plt.xlim([min_x - offset, max_x + offset])
    plt.ylim([min_y - offset, max_y + offset])

def plotData(real_data, fake_data, boundaries_real, boundaries_fake,
             recall_mask, coverage_mask, title, save, save_path):
    # Start plotting
    fig = plt.figure(figsize=(6, 10))
    fig.suptitle(title)
    # First subplot
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_title("Recall manifold")
    # Recall manifold
    plotCircles(fake_data, boundaries_fake)
    plt.scatter(fake_data[:, 0], fake_data[:, 1],
                label="Fake Samples", c="blue", s=2 ** 4, zorder=99, alpha=0.75)
    plotAcceptRejectData(real_data, recall_mask)
    setLimits(real_data, fake_data, boundaries_real, boundaries_fake)
    # Position relative to first subplot
    plt.legend(loc="lower left", ncol=1,
               prop={'size': 11})
    # setLimits(min_x, max_x, min_y, max_y)
    # Second subplot
    ax2 = plt.subplot(2, 1, 2, sharex=ax1, sharey=ax1)
    ax2.set_title("Coverage manifold")
    plt.scatter(fake_data[:, 0], fake_data[:, 1],
                label="Fake Samples", c="blue", s=2 ** 4, zorder=99, alpha=0.75)
    plotAcceptRejectData(real_data, coverage_mask)
    # Coverage manifold
    plotCircles(real_data, boundaries_real)

    if save:
        plt.subplots_adjust(wspace=0.3)
        plt.savefig(f"{save_path}points.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

def plotBox(data, xticks, title_text, save=False, save_path=""):
    plt.figure(figsize=(16, 6))
    plt.title(title_text)
    plt.boxplot(data)
    plt.xlabel("k_value")
    plt.xticks(np.arange(len(data)) + 1, xticks, rotation = 90)
    plt.ylabel("score")

    if save:
        plt.subplots_adjust(wspace=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
def dataframeBoxplot(score_dataframe, score_name, title_text):
    plt.ylim([0, 1.1])
    plt.title(title_text)
    sns.boxplot(x="k_val", y=score_name, data=score_dataframe).set(
        xlabel='k-value',
        ylabel=score_name
    )

def textBar(bars):
    for bar in bars:
        plt.annotate(f'{bar.get_height()}',
                     (bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01),
                     verticalalignment='bottom', horizontalalignment='center',
                     fontsize=9)
def plotBars(score_dataframe, score_dataframe_extra, score_name):
   score_original = score_dataframe[score_name].values
   score_extra = score_dataframe_extra[score_name].values
   k_vals = score_dataframe["k_val"].values
   x_vals = ((np.arange(score_original.shape[0]) + 1)*3)
   width = 1
   plt.ylim([0, 1.4])
   plt.xticks(x_vals, k_vals)
   bars = plt.bar(x_vals, score_original, -width, align="edge", label="Original fake data", color="black")
   bars2 = plt.bar(x_vals, score_extra, width, align="edge", label="Fake data with outliers", color="grey")
   textBar(bars)
   textBar(bars2)
   plt.legend()

def plotDistributions(real_data, fake_data, title_text, save_path, save=False):
    plt.title(title_text)
    plt.scatter(real_data[:, 0], real_data[:, 1], c="green", alpha=0.75, label="Real data")
    plt.scatter(fake_data[:, 0], fake_data[:, 1], c="red", alpha=1, label="Fake data")
    plt.legend()
    if save:
        plt.savefig(save_path)
        plt.close()


def plotCurve(curve, label_text):
    #plt.title(title_text)
    plt.xlim([0, 1.1])
    plt.ylim([0, 1.1])
    plt.scatter(curve[:, 1], curve[:, 0], label=label_text)
    plt.xlabel("Recall")
    plt.ylabel("Precision")

def specialAnnotate(text, coords, fontsize=12):
    plt.annotate(text, coords,
                 xytext=(10, -5), textcoords='offset points',
                 family='sans-serif', fontsize=fontsize, color='darkslategrey')
def plotAnnotate(series):
    for k, v in series.iterrows():
        text = v["k_val"]
        precision_recall = (v["recall"], v["precision"])
        density_coverage = (v["coverage"], v["density"])
        specialAnnotate(text, precision_recall)
        specialAnnotate(text, density_coverage)