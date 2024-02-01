import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Generate data for two distributions
x = np.linspace(-5, 5, 1000)
dist1 = np.exp(-x**2)
dist2 = np.exp(-(x-2)**2)

# Find intersection points
intersection = np.minimum(dist1, dist2)

# Find regions where distributions don't intersect
non_intersection_dist1 = np.where(dist1 > intersection, dist1, np.nan)
non_intersection_dist2 = np.where(dist2 > intersection, dist2, np.nan)

# Plot distributions
plt.plot(x, dist1, label='Distribution R')
plt.plot(x, dist2, label='Distribution F')



# Mark parts of distributions that don't intersect
plt.fill_between(x, non_intersection_dist1, color='blue', alpha=0.3, label='Loss of Recall')
plt.fill_between(x, non_intersection_dist2, color='orange', alpha=0.3, label='Loss of Precision')
# Highlight intersection region
plt.fill_between(x, intersection, color='gray', alpha=1, label='Intersection')
# Set labels and legend
plt.xlabel('X')
plt.ylabel('Probability Density')
plt.title('Distributions R and F with regions of interest')
plt.legend()

# Show plot
sub_map = "../example_figs/distributions/"
file_name = "distribution.png"
Path(sub_map).mkdir(parents=True, exist_ok=True)
save_path = f"{sub_map}/{file_name}"
plt.savefig(save_path, bbox_inches="tight", dpi=300)
plt.close()