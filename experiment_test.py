import matplotlib.pyplot as plt
import numpy as np
import experiments_v2.fake_diverse as experiment2
import experiments_v2.coverage_test as experiment3
import experiments_v2.uniform_test as experiment_uniform
import experiments_v2.helper_functions as helper
import visualize as plotting

def getExpectedRandomness(samples, weight):
    area = 4*weight
    randomness_score = 0.5*np.sqrt(area/samples)
    return randomness_score

def runUniform():
    iters = 1
    fake_samples = 100
    real_samples = 100
    fake_weight = [0.01, 0.25, 0.5, 0.75, 1]
    k_vals = [1, 4, 8]
    recall_tabular_data = np.zeros((len(fake_weight), len(k_vals)))
    coverage_tabular_data = np.zeros((len(fake_weight), len(k_vals)))
    saved_data = {"real_data": [], "fake_data": [], "k_val": [], "max_boundary": [], "titles": []}
    for column_id, k_val in enumerate(k_vals):
        for row_id, weight in enumerate(fake_weight):
            # real_data, fake_data = experiment_uniform.getData(start, high,
            #                                                   real_samples, fake_samples, weight)
            real_data, fake_data = experiment_uniform.getDataClusters(real_samples, fake_samples, weight)
            real_score = getExpectedRandomness(real_samples, 1)
            fake_score = getExpectedRandomness(fake_samples, weight)
            for i in range(iters):
                scores, avg_real_boundary, avg_fake_boundary = experiment_uniform.getScores(real_data, fake_data, k_val)
                rn_real = avg_real_boundary / real_score
                rn_fake = avg_fake_boundary / fake_score
                print(rn_real, rn_fake)
                recall = scores[1]
                coverage = scores[3]
                recall_tabular_data[row_id, column_id] = recall
                coverage_tabular_data[row_id, column_id] = coverage
                if i == 0:
                    saveData(saved_data, real_data, fake_data, k_val)
                    saved_data["titles"].append(f"Kval is {k_val} and {[rn_real, rn_fake]}")

    plotting.plotInterface(saved_data, save=False, save_path="./fig_v2/coverage/")


def saveData(saved_data, real_data, fake_data, k_val):
    saved_data["real_data"].append(real_data)
    saved_data["fake_data"].append(fake_data)
    saved_data["k_val"].append(k_val)
    boundaries_real, boundaries_fake, distance_matrix_pairs = helper.getBoundaries(real_data, fake_data, k_val)
    max_boundary = max(np.max(boundaries_real), np.max(boundaries_fake))
    saved_data["max_boundary"].append(max_boundary)


runUniform()
plt.show()