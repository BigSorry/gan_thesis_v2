import numpy as np
import metrics.metrics as mtr

def getReal(activation_dict, mode_samples, q=10):
    all_activations = []
    for mode in range(q):
        samples = activation_dict[mode][:mode_samples]
        all_activations.extend(samples)

    return all_activations

def getFake(real_features, modes, total_samples, q):
    fake_features = np.zeros((total_samples, real_features.shape[1]))
    mode_samples = fake_features // q
    class_samples = real_features.shape[0] // modes
    for mode in range(q):
        indices = np.random.choice(class_samples, mode_samples, replace=False)
        indices_mode = indices + (mode * class_samples)
        begin_index = mode_samples * mode
        end_index = begin_index + mode_samples
        taken_samples = real_features[begin_index:end_index, :]
        fake_features[begin_index:end_index, :] = taken_samples

    return fake_features

def modeDrop(activation_dict, fake_modes, method_name, params):
    mode_samples = 1000
    real_activations_sampled = getReal(activation_dict, mode_samples, q=5)
    fake_activations = getReal(activation_dict, mode_samples, q=fake_modes)
    pr_score, curve, cluster_labels = mtr.getPRCurve(real_activations_sampled, fake_activations, method_name, params)

    return pr_score, curve, cluster_labels