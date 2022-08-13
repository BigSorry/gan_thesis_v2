import experiments_pr.decrease_variances as exp_var
import experiments_pr.mean_shift as exp_mean
import experiments_pr.mode_drop as exp_drop
import numpy as np

class Experiment():
    def doExperiment(self, metric_name, metric_params):
        pass

    def initExperimentDict(self, experiment_vars):
        experiment_dict = {experiment_var:{} for experiment_var in experiment_vars}
        for experiment_var in experiment_vars:
            experiment_dict[experiment_var] = {}
            experiment_dict[experiment_var]["dim"] = []
            experiment_dict[experiment_var]["sample_count"] = []
            experiment_dict[experiment_var]["curve"] = []
            experiment_dict[experiment_var]["pr_score"] = []

        return experiment_dict

    def updateDict(self, experiment_dict, key, dimension,
                   sample_count, curve, pr_score):
        experiment_dict["experiments"][key]["dim"].append(dimension)
        experiment_dict["experiments"][key]["sample_count"].append(sample_count)
        experiment_dict["experiments"][key]["curve"].append(curve)
        experiment_dict["experiments"][key]["pr_score"].append(pr_score)

    def setSamples(self, samples_array):
        self.samples = samples_array

    def setDimensions(self, dim_array):
        self.dimension = dim_array

class MeanShift(Experiment):
    def doExperiment(self, metric_name, metric_params):
        mean_shifts = [0, 0.1, 0.25, 0.5, 0.75, 0.95, 1.25]
        result_dict = {"method": metric_name, "params": metric_params}
        result_dict["experiments"] = self.initExperimentDict(mean_shifts)
        for sample_count in self.samples:
            for dimension in self.dimension:
                for shift in mean_shifts:
                    pr_score, curve, cluster_labels = exp_mean.doShitfs(sample_count, dimension,
                              shift, metric_name, metric_params)
                    self.updateDict(result_dict, shift, dimension,
                               sample_count, curve, pr_score)

        return result_dict

class Covariance(Experiment):
    def doExperiment(self, metric_name, metric_params):
        zero_variance_ratios = [0, 0.25, 0.5, 0.75, .95]
        result_dict = {"method": metric_name, "params": metric_params}
        result_dict["experiments"] = self.initExperimentDict(zero_variance_ratios)
        for sample_count in self.samples:
            for dimension in self.dimension:
                for ratio in zero_variance_ratios:
                    pr_score, curve, cluster_labels = exp_var.subspaceExperimentOne(sample_count, dimension, ratio,
                                                                                    metric_name, metric_params)
                    self.updateDict(result_dict, ratio, dimension,
                                    sample_count, curve, pr_score)

        return result_dict

class ModeDrop(Experiment):
    def doExperiment(self, metric_name, metric_params):
        modes = 10
        start_concentration = 1 / modes
        mode_weights = np.linspace(start_concentration, 1, 10)
        result_dict = {"method": metric_name, "params": metric_params}
        result_dict["experiments"] = self.initExperimentDict(mode_weights)
        for sample_count in self.samples:
            for dimension in self.dimension:
                for mode_weight in mode_weights:
                    pr_score, curve, cluster_labels = exp_drop.setupExperiment(sample_count, dimension,
                                 mode_weight, modes, metric_name, metric_params)
                    self.updateDict(result_dict, mode_weight, dimension,
                                    sample_count, curve, pr_score)
        return result_dict

class ModeDropAssessing(Experiment):
    def doExperiment(self, metric_name, metric_params):
        modes = 10
        fake_modes = np.arange(10)+1
        result_dict = {"method": metric_name, "params": metric_params}
        result_dict["experiments"] = self.initExperimentDict(fake_modes)
        for sample_count in self.samples:
            for dimension in self.dimension:
                for fake_mode in fake_modes:
                    pr_score, curve, cluster_labels = exp_drop.modeDrop(sample_count, dimension,
                                 fake_mode, modes, metric_name, metric_params)
                    self.updateDict(result_dict, fake_mode, dimension,
                                    sample_count, curve, pr_score)
        return result_dict


