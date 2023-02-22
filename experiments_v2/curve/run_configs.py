import numpy as np
import union_experiment as exp

def checkScaling():
    samples=1000
    dimension = 2
    runs = 3
    var_factors = np.array([0.01, 0.1, 0.2, 0.25, 0.5, 0.75, 1]) / 2

    for i in range(runs):
        exp.doExperiment(sample_size=samples, dimension=dimension, lambda_factors=var_factors)
        var_factors *= 10

def runExperiment():
    sample_array = [1000, 3000, 5000]
    sample_array = [1000]
    dimensions = [2, 16, 32]
    var_factors = np.array([0.01, 0.1, 0.2, 0.25, 0.5, 0.75, 1])
    for dim in dimensions:
        for samples in sample_array:
            exp.doExperiment(samples, dim, var_factors)

runExperiment()