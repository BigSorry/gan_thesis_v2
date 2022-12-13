import numpy as np
import experiments_v2.helper_functions as util
import matplotlib.pyplot as plt

def createDistributions(params, sample_size, dimension, iters):
    functions = {}
    for param in params:
        for i in range(iters):
            samples = np.random.exponential(param, size=(sample_size, dimension))
            functions[param, i] = samples

    return functions

def getKL(base_param, other_param):
    return np.log(base_param) - np.log(other_param) + (other_param / base_param) - 1

def getCurve(distribution_dict, lambdas):
    curves = {}
    for base_param, base_data in distribution_dict.items():
        high_param = base_param[0]
        for other_key, other_data in distribution_dict.items():
            other_high_param = other_key[0]
            if high_param != other_high_param:
                pairs = []
                for val in lambdas:
                    intersection_lambda = val*high_param + (1-val)*other_high_param
                    base_kl = getKL(high_param, intersection_lambda)
                    other_kl = getKL(other_high_param, intersection_lambda)
                    pairs.append([base_kl, other_kl])

                curves[(high_param, other_high_param)] = np.array(pairs)

    return curves

def showData(data, data_other):
    plt.figure()
    y = np.zeros(data.shape[0])
    plt.scatter(data, y, s=2**7)
    plt.scatter(data_other, y, s=2**6)

def doExperiment():
    params = np.arange(1, 10, step=5)
    params = np.array([0.5, 0.6])
    lambdas = np.linspace(0, 1, num=100)
    sample_size=1000
    dimension=1
    iters = 1
    distribution_dict = createDistributions(params, sample_size, dimension, iters)
    curves_dict = getCurve(distribution_dict, lambdas)
    for param_key, curves in curves_dict.items():
        plt.figure()
        plt.title(param_key)
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.scatter(curves[:, 0], curves[:, 1])
        data = distribution_dict[(param_key[0], 0)]
        data_other = distribution_dict[(param_key[1], 0)]
        showData(data, data_other)
        break


doExperiment()
plt.show()