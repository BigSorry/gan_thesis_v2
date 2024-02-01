import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path

# Function to calculate distance between two points
def distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# Function to calculate kth nearest neighbor distance
def knn_distance(sample, samples, k):
    distances = [distance(sample, s) for s in samples]
    distances.sort()
    return distances[k]

# Define density function
def density(fake_samples, real_samples, k):
    density_score = 0
    for fake_sample in fake_samples:
        sample_count = sum(distance(fake_sample, real_sample) <= knn_distance(real_sample, real_samples,  k)
                           for real_sample in real_samples)
        density_score += sample_count

    return density_score / k / len(fake_samples)

# Define coverage function
def coverage(fake_samples, real_samples, k):
    coverage = 0
    for real_sample in real_samples:
        is_covered = any(
            distance(real_sample, fake_sample) <= knn_distance(real_sample, real_samples, k) for fake_sample in
            fake_samples)
        coverage += int(is_covered)
    return coverage / len(real_samples)

# Define precision and recall functions
def precision(fake_samples, real_samples, k):
    return sum(is_in_manifold(fake_sample, real_samples, k) for fake_sample in fake_samples) / len(fake_samples)

def recall(fake_samples, real_samples, k):
    return sum(is_in_manifold(real_sample, fake_samples, k) for real_sample in real_samples) / len(real_samples)

# Function to check if a sample falls into the manifold
def is_in_manifold(compare_sample, manifold_samples, k):
    return any(distance(compare_sample, manifold_sample) <= knn_distance(manifold_sample, manifold_samples, k) for manifold_sample in manifold_samples)

def plotCoverage(real_features, fake_features, k_value, title_text):
    # Plot manifolds for Density and Coverage
    for real_sample in real_features:
        is_covered = any(
            distance(real_sample, fake_sample) <= knn_distance(real_sample, real_features, k_value) for fake_sample in
            fake_features)
        circle = Circle(real_sample, knn_distance(real_sample, real_features, k_value), color='blue', fill=False)
        plt.gca().add_patch(circle)
        if is_covered:
            plt.annotate('Covered real sample', xy=real_sample,
                         xytext=(real_sample[0] + 0.05, real_sample[1] + 0.05),
                         arrowprops=dict(facecolor='black', shrink=0.05))
        else:
            plt.annotate('Not covered real sample', xy=real_sample,
                         xytext=(real_sample[0] + 0.05, real_sample[1] + 0.05),
                         arrowprops=dict(facecolor='white', shrink=0.05))
    plt.title(title_text)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
def plotManifolds(manifold_features, other_features, k_value, title_text, density_mode):
    # Plot manifolds for precision
    for sample in manifold_features:
        circle = Circle(sample, knn_distance(sample, manifold_features, k_value), color='blue', fill=False)
        plt.gca().add_patch(circle)

    # Plot additional samples and illustrate how they are counted
    for other_sample in other_features:
        if is_in_manifold(other_sample, manifold_features, k_value):
            if density_mode:
                density_count = sum(distance(other_sample, manifold_sample) <= knn_distance(manifold_sample, manifold_features, k_value) for manifold_sample in manifold_features)
                plt.annotate(f"Density count {density_count}", xy=other_sample,
                             xytext=(other_sample[0] + 0.05, other_sample[1] + 0.05),
                             arrowprops=dict(facecolor='black', shrink=0.05))
            else:
                plt.annotate('Counted as similar', xy=other_sample, xytext=(other_sample[0] + 0.05, other_sample[1] + 0.05),
                             arrowprops=dict(facecolor='black', shrink=0.05))
        else:
            #plt.scatter(other_sample[0], other_sample[1], color='red', marker='o', label="0 count")
            plt.annotate('Counted as dissimilar', xy=other_sample, xytext=(other_sample[0] + 0.05, other_sample[1] + 0.05),
                         arrowprops=dict(facecolor='white', shrink=0.05))
    # Plot settings
    plt.title(title_text)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()

def savePlots(map_path, file_name):
    Path(map_path).mkdir(parents=True, exist_ok=True)
    save_path = f"{map_path}/{file_name}"
    plt.savefig(save_path + ".pdf", bbox_inches="tight", format="pdf", dpi=150)
    plt.savefig(save_path + ".png", bbox_inches="tight", dpi=150)
    plt.close()
def prPlot(real_features, fake_features, k_value, map_path):
    recall_score = recall(fake_features, real_features, k_value)
    precision_score = precision(fake_features, real_features, k_value)
    print(precision_score, recall_score)
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 2)
    plt.scatter(real_features[:, 0], real_features[:, 1], color='green', label="Real Samples")
    plt.scatter(fake_features[:, 0], fake_features[:, 1], color='red', label="Fake Samples")

    plotManifolds(fake_features, real_features, k_value, f"Recall {recall_score}", density_mode=False)
    plt.subplot(1, 2, 1)
    plt.scatter(real_features[:, 0], real_features[:, 1], color='green', label="Real Samples")
    plt.scatter(fake_features[:, 0], fake_features[:, 1], color='red', label="Fake Samples")
    plotManifolds(real_features, fake_features, k_value, f"Precision {precision_score}", density_mode=False)

    savePlots(map_path, f"example_pr_{iter}")

def dcPlot(real_features, fake_features, k_value, map_path):
    coverage_score = coverage(fake_features, real_features, k_value)
    density_score = density(fake_features, real_features, k_value)
    print(density_score, coverage_score)
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 2)
    plt.scatter(real_features[:, 0], real_features[:, 1], color='green', label="Real Samples")
    plt.scatter(fake_features[:, 0], fake_features[:, 1], color='red', label="Fake Samples")
    plotCoverage(real_features, fake_features, k_value, f"Coverage {coverage_score}")
    plt.subplot(1, 2, 1)
    plt.scatter(real_features[:, 0], real_features[:, 1], color='green', label="Real Samples")
    plt.scatter(fake_features[:, 0], fake_features[:, 1], color='red', label="Fake Samples")
    plotManifolds(real_features, fake_features, k_value, f"Density {density_score}", density_mode=True)
    savePlots(map_path, f"example_dc_{iter}")

def doPlots(iter_nr):
    k_value = 1
    samples = 5
    fake_features = np.random.rand(samples, 2)
    real_features = np.random.rand(samples, 2)
    map_path = "../experiment_figures/examples/"
    prPlot(real_features, fake_features, k_value, map_path)
    dcPlot(real_features, fake_features, k_value, map_path)

iters = 10
for iter in range(iters):
    print(iter)
    doPlots(iter)
    print()
