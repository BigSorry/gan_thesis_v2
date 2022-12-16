import numpy as np
import pandas as pd
import experiments_v2.helper_functions as util
import matplotlib.pyplot as plt
import metrics.not_used.likelihood_classifier as llc
import visualize as plotting

def getDataframe(distribution_dict, k_vals):
    columns = ["key", "k_val",
               "precision", "recall", "density", "coverage"]
    row_data = []
    for base_param, base_data in distribution_dict.items():
        high_param = base_param[1]
        for other_key, other_data in distribution_dict.items():
            other_high_param = other_key[1]
            distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = util.getDistanceMatrices(base_data, other_data)
            for k_val in k_vals:
                boundaries_real = distance_matrix_real[:, k_val]
                boundaries_fake = distance_matrix_fake[:, k_val]
                precision, recall, density, coverage = util.getScores(distance_matrix_pairs, boundaries_fake,
                                                              boundaries_real, k_val)
                row = [(high_param, other_high_param), k_val,
                       precision, recall, density, coverage]
                row_data.append(row)

    dataframe = pd.DataFrame(columns=columns, data=row_data)
    return dataframe

def createDistributions(params, sample_size, dimension, iters):
    functions = {}
    for param in params:
        for i in range(iters):
            samples = np.random.exponential(param, size=(sample_size, dimension))
            functions[param, i] = samples

    return functions

def getLikelihood(real_features, fake_features, real_param, fake_param, lambdas):
    mixture_samples = np.concatenate([real_features, fake_features])
    mixture_labels = np.concatenate([np.ones(real_features.shape[0]), np.zeros(fake_features.shape[0])])
    pr_curve, differences = llc.getPRCurve(mixture_samples, mixture_labels, lambdas, real_param,
                              fake_param, distribution_name="exp")

    return pr_curve, differences

def getPRLambdas(angle_count = 50):
    epsilon = 1e-10
    angles = np.linspace(epsilon, np.pi / 2 - epsilon, num=angle_count)
    lambdas = np.tan(angles)

    return lambdas

def plotValues(x,y, title_text):
    plt.title(title_text)
    plt.plot(x, y)
    plt.xlabel("Lambda")
    plt.ylabel("Mean absolute differences pdfs")
    plt.ylim([0, np.max(y)+1])

def doPlots(base_data, other_data, param_text, curves, differences, lambdas, title_text):
    plt.figure()
    plt.subplot(1, 3, 1)
    plotting.plotCurve(curves, title_text)
    plt.subplot(1, 3, 2)
    plotValues(lambdas, differences, f"Exp distributions {title_text}")
    plt.subplot(1 ,3, 3)
    plotting.plotDistributions(base_data, other_data, param_text)

def doExperiment():
    params = np.arange(1, 102, step=25)
    params = np.array([1, 99])
    print(params)
    pr_lambdas = getPRLambdas(angle_count=1000)
    k_vals = [1, 2, 4, 8]

    sample_size=1000
    dimension=1
    iters = 1
    distribution_dict = createDistributions(params, sample_size, dimension, iters)
    dataframe = getDataframe(distribution_dict, k_vals)
    images = 0
    max_images = 100
    for base_param, base_data in distribution_dict.items():
        first_param = base_param[0]
        for other_param, other_data in distribution_dict.items():
            second_param = other_param[0]
            param_text  = f"{first_param}_{second_param}"
            if images < max_images and first_param != second_param:
                curves, differences = getLikelihood(base_data, other_data, first_param, second_param, pr_lambdas)
                doPlots(base_data, other_data, param_text,
                        curves, differences, pr_lambdas, param_text)

                images+=1




doExperiment()
plt.show()