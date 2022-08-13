import experiments_pr.activation_experiment as act_exp
import experiment_object as object
import numpy as np

class ModeDropActivation(object.Experiment):
    def setActivations(self, activations):
        self.activations = activations

    def doExperiment(self, metric_name, metric_params):
        fake_modes = np.arange(10)+1
        result_dict = {"method": metric_name, "params": metric_params}
        result_dict["experiments"] = self.initExperimentDict(fake_modes)
        for fake_mode in fake_modes:
            pr_score, curve, cluster_labels = act_exp.modeDrop(self.activations, fake_mode, metric_name, metric_params)
            self.updateDict(result_dict, fake_mode, self.dimension,
                            self.samples, curve, pr_score)

        return result_dict


