import numpy as np
import experiments_v2.helper_functions as util
import matplotlib.pyplot as plt
import metrics.not_used.likelihood_classifier as llc
import visualize as plotting

def createDistributions(params, sample_size, dimension, iters):
    functions = {}
    for param in params:
        for i in range(iters):
            samples = np.random.exponential(param, size=(sample_size, dimension))
            functions[param, i] = samples

    return functions

def getKL(base_param, other_param):
    return np.log(base_param) - np.log(other_param) + (other_param / base_param) - 1

def getRenyi(base_param, other_param, alpha):
    base_part = alpha*(np.log(base_param) - (base_param / base_param))
    other_part = (alpha-1)*(np.log(other_param) - (other_param / base_param))
    alpha_part = 1 / (alpha-1)
    result = alpha_part*(base_part - other_part)

    return result
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
                    base_kl = getRenyi(high_param, intersection_lambda, alpha=9999)
                    other_kl = getRenyi(other_high_param, intersection_lambda, alpha=9999)
                    pairs.append([base_kl, other_kl])

                curves[(high_param, other_high_param)] = np.array(pairs)

    return curves

def getLikelihood(real_features, fake_features, real_param, fake_param, lambdas):
    mixture_samples = np.concatenate([real_features, fake_features])
    mixture_labels = np.concatenate([np.ones(real_features.shape[0]), np.zeros(fake_features.shape[0])])
    pr_curve = llc.getPRCurve(mixture_samples, mixture_labels, lambdas, real_param,
                              fake_param, distribution_name="exp")

    return pr_curve
def getPRLambdas(angle_count = 50):
    epsilon = 1e-10
    angles = np.linspace(epsilon, np.pi / 2 - epsilon, num=angle_count)
    lambdas = np.tan(angles)

    return lambdas

def doExperiment():
    params = np.arange(0., 20, step=5)
    params = [0.5, 2]
    pr_lambdas = getPRLambdas(angle_count=100)
    divergence_lambdas = pr_lambdas[pr_lambdas <= 1]
    sample_size=1000
    dimension=1
    iters = 1
    distribution_dict = createDistributions(params, sample_size, dimension, iters)
    curves_dict = getCurve(distribution_dict, divergence_lambdas)

    for param_key, curves in curves_dict.items():
        first_param = param_key[0]
        second_param = param_key[1]
        plotting.plotCurve(curves, param_key)
        data = distribution_dict[(param_key[0], 0)]
        data_other = distribution_dict[(param_key[1], 0)]
        curves = getLikelihood(data, data_other, first_param, second_param, pr_lambdas)
        plotting.plotCurve(curves, param_key)
        plotting.plotDistributions(data, data_other)





doExperiment()
plt.show()