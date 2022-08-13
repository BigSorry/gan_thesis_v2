import experiment_object as exp_obj
import experiment_activation as exp_act

class CreateExperiment:
    def getExperiment(self, experiment_id):
        if experiment_id == 1:
            return exp_obj.MeanShift()
        elif experiment_id == 2:
            return exp_obj.Covariance()
        elif experiment_id == 3:
            return exp_obj.ModeDrop()
        elif experiment_id == 4:
            return exp_obj.ModeDropAssessing()
        elif experiment_id == 5:
            return exp_act.ModeDropActivation()