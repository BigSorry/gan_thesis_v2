import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
def HeatMapPivot(pivot_table, title_text="", save=False, save_path=""):
    plt.title(title_text)
    ax = sns.heatmap(pivot_table, cmap="RdYlGn_r", annot=True,
                annot_kws={"color": "black", "backgroundcolor": "white"},
                     vmin=0, vmax=1)
    plt.yticks(rotation=0)
    ax.invert_yaxis()
    if save:
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def plotTheoreticalCurve(curve_classifier, curve_var_dist, scale_factors, save=True):
    #plt.title(f"Lambda scaling real cov {scale_factors[0]} and lambda scaling fake cov {scale_factors[1]}")
    #plotCurve(curve_classifier, label_text="Likelihood ratio test")
    plotCurve(curve_var_dist, label_text="Variational distance")
    plt.legend()
    if save:
        path = f"C:/Users/lexme/Documents/gan_thesis_v2/present/1-02-23/ground-truths/scale_{scale_factors}.png"
        plt.savefig(path)

# Plotting is reversed to get recall on x axis
def plotKNNMetrics(score_pair,  k_values, label_name, color, save_path, save=True):
    annotate_text = [f"k={k}" for k in k_values]
    plt.scatter(score_pair[:, 1], score_pair[:, 0], c=color, label=label_name)
    for index, text in enumerate(annotate_text):
        coords = (score_pair[index, 1], score_pair[index, 0])
        specialAnnotate(text, coords, fontsize=14)
    if save:
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   fancybox=True, shadow=True, ncol=2, fontsize=9)
        plt.savefig(save_path)
        plt.close()

def plotStats(pr_above, dc_above, save_path, save=False):
    pr_mean = pr_above.mean()
    dc_mean = dc_above.mean()
    ax = ["pr", "dc"]
    plt.bar(ax, [pr_mean, dc_mean], color='black', width=0.25)
    plt.ylim([0, 1.1])
    plt.xlabel("Percentage points above the curve")
    if save:
        plt.savefig(save_path)
        plt.close()

def plotCurveMetrics(histo_method, classifier_method, scale_factors, save=True):
    plt.scatter(histo_method[:, 1], histo_method[:, 0], c="green", label=f"Precision_Recall_Histo_Curve")
    plt.scatter(classifier_method[:, 1], classifier_method[:, 0], c="black", label=f"Precision_Recall_Class_Curve")
    plt.legend()
    if save:
        path = f"C:/Users/lexme/Documents/gan_thesis_v2/plot_paper/gaussian/scale_{scale_factors}.png"
        plt.savefig(path)

def plotDistributions(real_data, fake_data, r_order, f_order, title_text, save_path, save=False):
    plt.title(title_text)
    plt.scatter(real_data[:, 0], real_data[:, 1], c="green", zorder=r_order, label="Real data")
    plt.scatter(fake_data[:, 0], fake_data[:, 1], c="red", zorder=f_order, label="Fake data")
    plt.legend(loc="lower right")
    if save:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def plotBars(k_vals, score_pairs, first_name, second_name):
   first_score = score_pairs[:, 0]
   second_score = score_pairs[:, 1]
   x_vals = ((np.arange(first_score.shape[0]) + 1)*3)
   width = 1
   plt.ylim([0, 1.1])
   plt.xticks(x_vals, k_vals)
   plt.xlabel("k-value")
   bars = plt.bar(x_vals, first_score, -width, align="edge", label=f"{first_name}", color="black")
   bars2 = plt.bar(x_vals, second_score, width, align="edge", label=f"{second_name}", color="grey")
   # textBar(bars)
   # textBar(bars2)
   plt.legend()

def textBar(bars):
    for bar in bars:
        plt.annotate(f'{bar.get_height()}',
                     (bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01),
                     verticalalignment='bottom', horizontalalignment='center',
                     fontsize=9)

def plotLine(x, y, label_text=""):
    plt.ylim([0,1.1])
    plt.xlabel("k-value")
    plt.plot(x, y, label=label_text)

def plotCurve(curve, label_text):
    # plt.title(title_text)
    plt.xlim([0, 1.1])
    plt.ylim([0, 1.1])
    plt.scatter(curve[:, 1], curve[:, 0], label=label_text)
    plt.xlabel("Recall")
    plt.ylabel("Precision")

def boxPlot(title_text, xticks, y_values, save=False, save_path=""):
    plt.title(title_text)
    plt.xlabel('Dimensions')
    plt.ylabel('Distance')
    plt.ylim([0, 1.1])
    plt.boxplot(y_values)
    plt.xticks(np.arange(len(xticks)) + 1, xticks, rotation = 90)
    if save:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
def plotErrorbar(title_text, x, means, stds, xlim=[], save=False, save_path=""):
    plt.title(title_text)
    plt.xlabel('Dimensions')
    plt.ylabel('Distance')
    plt.ylim([0, 1.1])
    plt.xlim(xlim)
    plt.errorbar(x, means, stds, linestyle='None', marker='o', color='blue')
    if save:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def specialAnnotate(text, coords, fontsize=12):
    plt.annotate(text, coords,
                 xytext=(10, -5), textcoords='offset points',
                 family='sans-serif', fontsize=fontsize, color='darkslategrey')

def saveLambaBoxplot(dataframe, score_name, map_path):
    dimensions = dataframe["dimension"].unique()
    lambda_factors = dataframe["lambda_factor"].unique()
    score_map = f"{map_path}/{score_name}/"
    if not os.path.exists(score_map):
        os.makedirs(score_map)
    for dim in dimensions:
        for scale in lambda_factors:
            save_path = f"{score_map}/dim{dim}_lambda{scale}.png"
            sel_data = dataframe.loc[(dataframe["dimension"] == dim) & (dataframe["lambda_factor"] == scale), :]
            grouped = sel_data.groupby(["k_val"]).agg([np.mean, np.std]).reset_index()
            score_means = grouped[score_name]["mean"]
            score_std = grouped[score_name]["std"]
            k_vals = grouped["k_val"]
            plt.figure(figsize=(14, 6))
            plt.errorbar(k_vals, score_means, score_std, linestyle='None', marker='o')
            plt.ylim([0, 1.1])
            plt.xlabel("K-value")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()

def saveLambaBoxplotCombine(dataframe, score_name, map_path, factors):
    dimensions = dataframe["dimension"].unique()
    for dim in dimensions:
        save_path = f"{map_path}/{score_name}_dim{dim}_{factors}.png"
        sel_data = dataframe.loc[(dataframe["dimension"] == dim), :]
        grouped = sel_data.groupby(["k_val"]).agg([np.mean, np.std]).reset_index()
        score_means = grouped[score_name]["mean"]
        score_std = grouped[score_name]["std"]
        k_vals = grouped["k_val"]
        plt.figure(figsize=(14, 6))
        plt.errorbar(k_vals, score_means, score_std, linestyle='None', marker='o')
        plt.ylim([0, 1.1])
        plt.xlabel("K-value")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def saveLambaBoxplotDimensions(dataframe, score_name, map_path, factors):
    plt.figure(figsize=(14, 6))
    sample_sizes = dataframe["sample_size"].unique()
    dimensions = dataframe["dimension"].unique()
    for sample_size in sample_sizes:
        for dim in dimensions:
            sel_data = dataframe.loc[(dataframe["dimension"] == dim) & (dataframe["sample_size"] == sample_size), :]
            grouped = sel_data.groupby(["k_val"]).agg([np.mean, np.std]).reset_index()
            score_means = grouped[score_name]["mean"]
            score_std = grouped[score_name]["std"]
            k_vals = grouped["k_val"]
            plt.errorbar(k_vals, score_means, score_std, linestyle='None', marker='o', label=f"dim_{dim}")

        plt.ylim([0, 1.1])
        plt.xlabel("K-value")
        plt.legend()
        save_path = f"{map_path}/{score_name}_s{sample_size}_{factors}.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()