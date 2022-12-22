import numpy as np
import pandas as pd
import experiments_v2.helper_functions as util
import matplotlib.pyplot as plt
import metrics.likelihood_classifier as llc
import visualize as plotting

def getDataframe(distribution_dict, k_vals):
    columns = ["key", "k_val",
               "precision", "recall", "density", "coverage"]
    row_data = []
    for base_param, base_data in distribution_dict.items():
        high_param = base_param[0]
        for other_key, other_data in distribution_dict.items():
            other_high_param = other_key[0]
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

def createDistributions(param_dict, sample_size, dimension, iters):
    functions = {}
    for param in param_dict["params"]:
        for i in range(iters):
            if param_dict["distribution"] == "exp":
                samples = np.random.exponential(param, size=(sample_size, dimension))
                functions[param, i] = samples
            elif param_dict["distribution"] == "gaus":
                mean_vec = param[0]
                cov = param[1]
                samples = np.random.multivariate_normal(mean_vec, cov, sample_size)
                functions[cov[0,0], i] = samples

    return functions

def getLikelihood(real_features, fake_features, real_param, fake_param, lambdas, distribution_name):
    mixture_samples = np.concatenate([real_features, fake_features])
    mixture_labels = np.concatenate([np.ones(real_features.shape[0]), np.zeros(fake_features.shape[0])])
    #mixture_samples, mixture_labels = llc.getMixture(real_features, fake_features)
    pr_curve, differences = llc.getPRCurve(mixture_samples, mixture_labels, lambdas, real_param,
                              fake_param, distribution_name=distribution_name)

    return pr_curve, differences

def getPRLambdas(angle_count = 50):
    epsilon = 1e-10
    angles = np.linspace(epsilon, np.pi / 2 - epsilon, num=angle_count)
    lambdas = np.tan(angles)

    return lambdas

def doPlots(dataframe, base_data, other_data, param_text, curves, title_text):
    plt.figure()
    plt.subplot(1, 2, 1)
    plotting.plotCurve(curves, title_text)
    plt.scatter(dataframe["recall"], dataframe["precision"], c="red", label="Precision_Recall")
    plt.scatter(dataframe["coverage"], dataframe["density"], c="yellow", label="Density_Coverage")
    plotting.plotAnnotate(dataframe)
    plt.legend()
    plt.subplot(1 ,2, 2)
    plotting.plotDistributions(base_data[:,0], base_data[:, 1], other_data[:, 0],
                               other_data[:, 1], param_text)

def getGausParams(scale_factors, dimension):
    gaus_params = []
    for scale in scale_factors:
        mean = np.zeros(dimension)
        cov = np.eye(dimension) * scale
        gaus_params.append([mean, cov])

    return gaus_params
def doExperiment():
    # Experiment params
    sample_size = 1000
    dimension = 2
    iters = 1
    pr_lambdas = getPRLambdas(angle_count=1000)
    k_vals = [1, 7, sample_size-1]
    # Distribution parameters
    exp_params = [0.5, 1.5]
    scale_factors = [0.5,  1]
    gaus_param = getGausParams(scale_factors, dimension)
    param_dict = {"distribution": "gaus", "params": gaus_param}
    #param_dict = {"distribution": "exp", "params": exp_params}
    # Prep data
    distribution_dict = createDistributions(param_dict, sample_size, dimension, iters)
    dataframe = getDataframe(distribution_dict, k_vals)
    images = 0
    max_images = 2
    for base_param, base_data in distribution_dict.items():
        first_param = base_param[0]
        for other_param, other_data in distribution_dict.items():
            second_param = other_param[0]
            param_text = f"{first_param}_{second_param}"
            if images < max_images and first_param != second_param:
                curves, differences = getLikelihood(base_data, other_data, first_param, second_param, pr_lambdas, param_dict["distribution"])
                doPlots(dataframe, base_data, other_data, param_text,
                        curves, param_text)

                images+=1




doExperiment()
plt.show()