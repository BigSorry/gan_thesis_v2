import numpy as np
import matplotlib.pyplot as plt
from experiments import create_experiment as exp

def getData():
    sample_sizes = [2, 4, 8, 16, 32, 64, 512, 1024]
    dimensions = [2, 4, 8, 16, 32, 64, 512, 1024]
    k_vals = np.array([1, 3, 7, 9, 16, 32, 64, num_real_samples - 1])
    for dimension in dimensions:
        real_features = np.random.normal(loc=0.0, scale=1.0,
                                         size=[num_real_samples, dimension])

        fake_features = np.random.normal(loc=0.0, scale=1.0,
                                         size=[num_fake_samples, dimension])

        map_path = f"./gaussian_curve/"
        pr_aboves, dc_aboves, pr_nearest_distances, dc_nearest_distances = exp.doExperiment("gaussian",
                                                                                            real_features,
                                                                                            fake_features,
                                                                                            1, 1,
                                                                                            k_vals,
                                                                                            save_curve=True,
                                                                                            map_path=map_path)

    plt.show()

getData()