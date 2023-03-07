import matplotlib.pyplot as plt

def plotTheoreticalCurve(curve_classifier, curve_var_dist, scale_factors, save=True):
    #plt.title(f"Lambda scaling real cov {scale_factors[0]} and lambda scaling fake cov {scale_factors[1]}")
    #plotCurve(curve_classifier, label_text="Likelihood ratio test")
    plotCurve(curve_var_dist, label_text="Variational distance")
    plt.legend()
    if save:
        path = f"C:/Users/lexme/Documents/gan_thesis_v2/present/1-02-23/ground-truths/scale_{scale_factors}.png"
        plt.savefig(path)

# Plotting is reversed to get recall on x axis
def plotKNNMetrics(pr_pairs, dc_pairs, pr_above, dc_above, k_values, save_path, save=True):
    annotate_text = [f"k={k}" for k in k_values]
    plt.scatter(pr_pairs[:, 1], pr_pairs[:, 0], c="yellow", label=f"Precision_Recall_KNN")
    #plt.scatter(pr_pairs[pr_above, 1], pr_pairs[pr_above, 0], c="green", label=f"Upper_Precision_Recall_KNN")
    #plt.scatter(dc_pairs[~dc_above, 1], dc_pairs[~dc_above, 0], c="yellow", label=f"Under_Density_Coverage_KNN")
    plt.scatter(dc_pairs[:, 1], dc_pairs[:, 0], c="black", label=f"Density_Coverage_KNN")
    for index, text in enumerate(annotate_text):
        pr_coords = (pr_pairs[index, 1], pr_pairs[index, 0])
        dc_coords = (dc_pairs[index, 1], dc_pairs[index, 0])
        specialAnnotate(text, pr_coords, fontsize=14)
        specialAnnotate(text, dc_coords, fontsize=14)
    plt.legend(loc='upper center', bbox_to_anchor=(1, 1.15),
               fancybox=True, shadow=True, ncol=2, fontsize=9)
    if save:
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

def plotDistributions(real_data, fake_data, title_text, save_path, save=False):
    plt.title(title_text)
    plt.scatter(real_data[:, 0], real_data[:, 1], c="green", alpha=0.75, label="Real data")
    plt.scatter(fake_data[:, 0], fake_data[:, 1], c="red", alpha=1, label="Fake data")
    plt.legend(loc="lower right")
    if save:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def plotCurve(curve, label_text):
    # plt.title(title_text)
    plt.xlim([0, 1.1])
    plt.ylim([0, 1.1])
    plt.scatter(curve[:, 1], curve[:, 0], label=label_text)
    plt.xlabel("Recall")
    plt.ylabel("Precision")


def specialAnnotate(text, coords, fontsize=12):
    plt.annotate(text, coords,
                 xytext=(10, -5), textcoords='offset points',
                 family='sans-serif', fontsize=fontsize, color='darkslategrey')
