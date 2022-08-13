import numpy as np
import metrics.metrics as mtr
import experiment_utils as util
import visualize as plotting

def getData(total_samples, mean_vectors, cov_matrix):
    modes = mean_vectors.shape[0]
    dimension = mean_vectors.shape[1]
    real_features = np.zeros((total_samples, dimension))
    mode_samples = total_samples // modes
    for mode in range(modes):
        mean_vec = mean_vectors[mode, :]
        mode_features = np.random.multivariate_normal(mean_vec, cov_matrix, size=mode_samples)
        begin_index = mode_samples * mode
        real_features[begin_index:begin_index+mode_samples, :] = mode_features

    return real_features

def getFake(total_samples, mean_vectors, cov_matrix, mode_weight):
    modes = mean_vectors.shape[0]
    first_mode_samples = total_samples * mode_weight
    other_mode_samples = (total_samples-first_mode_samples) // (modes-1)
    fake_mode_samples = [other_mode_samples for i in range(modes)]
    fake_mode_samples[0] = first_mode_samples
    fake_mode_samples = np.int32(fake_mode_samples)
    fake_features = []
    for index, mode_samples in enumerate(fake_mode_samples):
        mean_vec = mean_vectors[index, :]
        mode_features = np.random.multivariate_normal(mean_vec, cov_matrix, size=mode_samples)
        fake_features.extend(mode_features)

    return np.array(fake_features), fake_mode_samples


def saveProblem(path):
    modes = 10
    mode_samples = 1000
    dims = [2, 4, 8, 16, 32, 64]
    dims = [2, 64]
    mode_weights = [0.01, 0.1, 0.25, 0.5, 0.75, 1]

    problem_dict = {dim: {} for dim in dims}
    for dimension in dims:
        covariance = np.eye(dimension)
        scalars = np.random.uniform(-1, 1, size=(modes, dimension))
        mean_vectors = 1 * scalars
        mode_features = getData(mode_samples, mean_vectors, covariance)

        problem_dict[dimension]["real"] = mode_features
        problem_dict[dimension]["modes"] = modes
        problem_dict[dimension]["mode_samples"] = mode_samples
        problem_dict[dimension]["weights"] = mode_weights
        problem_dict[dimension]["mean_vectors"] = mean_vectors
        problem_dict[dimension]["covariance"] = covariance

    util.savePickle(path, problem_dict)

def getParams(dimension, modes):
    covariance = np.eye(dimension)
    scalars = np.random.uniform(-1, 1, size=(modes, dimension))
    mean_vectors = 1 * scalars

    return mean_vectors, covariance


def setupExperiment(sample_count, dimension, mode_weight,
                    modes, method_name, params):
    mean_vectors, cov_matrix = getParams(dimension, modes)
    real_features = getData(sample_count, mean_vectors, cov_matrix)
    fake_features, fake_mode_samples = getFake(sample_count, mean_vectors, cov_matrix, mode_weight)
    print(fake_mode_samples, mode_weight)
    pr_score, curve, cluster_labels = mtr.getPRCurve(real_features, fake_features, method_name, params)

    return pr_score, curve, cluster_labels


def getReal2(total_samples, mean_vectors, cov_matrix, q=5):
    dimension = mean_vectors.shape[1]
    real_features = np.zeros((total_samples, dimension))
    mode_samples = total_samples // q
    for mode in range(q):
        mean_vec = mean_vectors[mode, :]
        mode_features = np.random.multivariate_normal(mean_vec, cov_matrix, size=mode_samples)
        begin_index = mode_samples * mode
        real_features[begin_index:begin_index + mode_samples, :] = mode_features

    return real_features

def getFake2(total_samples, mean_vectors, cov_matrix, q=5):
    modes = mean_vectors.shape[0]
    dimension = mean_vectors.shape[1]
    fake_features = np.zeros((total_samples, dimension))
    mode_samples = total_samples // q
    for mode in range(q):
        mean_vec = mean_vectors[mode, :]
        mode_features = np.random.multivariate_normal(mean_vec, cov_matrix, size=mode_samples)
        begin_index = mode_samples * mode
        fake_features[begin_index:begin_index + mode_samples, :] = mode_features

    return fake_features

def modeDrop(sample_count, dimension, fake_modes, modes, method_name, params):
    mean_vectors, cov_matrix = getParams(dimension, modes)
    real_features = getReal2(sample_count, mean_vectors, cov_matrix, q=5)
    fake_features = getFake2(sample_count, mean_vectors, cov_matrix, q=fake_modes)
    pr_score, curve, cluster_labels = mtr.getPRCurve(real_features, fake_features, method_name, params)

    return pr_score, curve, cluster_labels

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import KFold
def testProblem(problem_dict):
    for dim, dim_dict in problem_dict.items():
        real_samples = dim_dict["real"]
        modes = dim_dict["modes"]
        modes_samples = dim_dict["mode_samples"]
        real_labels = np.zeros(modes*modes_samples, dtype=np.int)
        for i in range(modes):
            begin_index = i * modes_samples
            real_labels[begin_index:begin_index+modes_samples] = np.ones(modes_samples)*i

        kf = KFold(n_splits=5)
        shuffle_indices = np.random.choice(real_samples.shape[0], real_samples.shape[0], replace=False)
        shuffled_samples = real_samples[shuffle_indices, :]
        shuffled_labels = real_labels[shuffle_indices]
        print(dim)
        errors = []
        for train, test in kf.split(shuffled_samples):
            train_samples = shuffled_samples[train, :]
            test_samples = shuffled_samples[test, :]
            train_labels = shuffled_labels[train]
            test_labels = shuffled_labels[test]

            clf = RandomForestClassifier().fit(train_samples, train_labels)
            predicted_labels = clf.predict(test_samples)
            error_rate = np.mean(predicted_labels != test_labels)
            errors.append(error_rate)
        print(errors)
        print(np.mean(errors))

