from typing import List, Union, Dict

import torch
from torch import Tensor

from avalanche.evaluation import Metric, PluginMetric, GenericPluginMetric
from avalanche.evaluation.metrics.mean import Mean
from avalanche.evaluation.metrics import Accuracy, TaskAwareAccuracy
from collections import defaultdict


class EnsembleAccuracy(Accuracy):
    def __init__(self):
        self._mean_accuracy = Mean()

    @torch.no_grad()
    def update(self, predicted_y: Tensor, true_y: Tensor):
        predicted_y = predicted_y.mean(dim=0) # Average over mc_samples
        super().update(predicted_y, true_y)     

    def result(self) -> float:
        return self._mean_accuracy.result()
    
    def reset(self) -> None:
        self._mean_accuracy.reset()

class TaskAwareEnsembleAccuracy(EnsembleAccuracy):
    def __init__(self):
        self._mean_accuracy = defaultdict(EnsembleAccuracy)

    @torch.no_grad()
    def update(self, predicted_y: Tensor, true_y: Tensor, task_id: Tensor):
        if isinstance(task_id, int):
            self._mean_accuracy[task_id].update(predicted_y, true_y)
        elif isinstance(task_id, Tensor):
            for pred, true, task in zip(predicted_y, true_y, task_id):
                if isinstance(task, Tensor):
                    task = task.item()
                self._mean_accuracy[task].update(pred.unsqueeze(0), true.unsqueeze(0))
        else:
            raise ValueError(
                f"Task label type: {type(task_id)}, "
                f"expected int/float or Tensor"
            )
        
    def result(self, task_label=None) -> Dict[int, float]:
        assert task_label is None or isinstance(task_label, int)
        if task_label is None:
            return {
                task_label: self._mean_accuracy[task_label].result()
                for task_label in self._mean_accuracy
            }
        else:
            return {task_label: self._mean_accuracy[task_label].result()}

    def reset(self, task_label=None) -> None:
        assert task_label is None or isinstance(task_label, int)
        if task_label is None:
            self._mean_accuracy = defaultdict(EnsembleAccuracy)
        else:
            self._mean_accuracy[task_label].reset()

class EnsembleAccuracyPluginMetric(GenericPluginMetric[float]):
    def __init__(self, reset_at, emit_at, mode, split_by_task=False):
        self.split_by_task = split_by_task
        if self.split_by_task:
            self._accuracy = TaskAwareEnsembleAccuracy()
        else:
            self._accuracy = EnsembleAccuracy()
        super(EnsembleAccuracyPluginMetric, self).__init__(
            self._accuracy, reset_at=reset_at, emit_at=emit_at, mode=mode
        )

    def update(self, strategy):
        if isinstance(self._accuracy, EnsembleAccuracy):
            self._accuracy.update(strategy.mb_ensem_outputs, strategy.mb_y)
        elif isinstance(self._accuracy, TaskAwareEnsembleAccuracy):
            self._accuracy.update(
                strategy.mb_ensem_outputs, strategy.mb_y, strategy.mb_task_id
            )
        else:
            assert False, "should never get here."

    def result(self, strategy):
        return self._metric.result()
    
    def reset(self, strategy):
        self._metric.reset()
    
class ExperienceEnsembleAccuracy(EnsembleAccuracyPluginMetric):
    def __init__(self):
        super(ExperienceEnsembleAccuracy, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval"
        )

    def __str__(self):
        return "Ensemble_Acc_Exp"
    
class StreamEnsembleAccuracy(EnsembleAccuracyPluginMetric):
    def __init__(self):
        super(StreamEnsembleAccuracy, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval"
        )

    def __str__(self):
        return "Ensemble_Acc_Stream"
    
class TrainedExperienceEnsembleAccuracy(EnsembleAccuracyPluginMetric):
    def __init__(self):
        super(TrainedExperienceEnsembleAccuracy, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval",
        )
        self._current_experience=0

    def after_training_exp(self, strategy):
        self._current_experience = strategy.experience.current_experience
        EnsembleAccuracyPluginMetric.reset(self, strategy)
        return EnsembleAccuracyPluginMetric.after_training_exp(self, strategy)
    
    def update(self, strategy):
        if strategy.experience.current_experience <= self._current_experience:
            EnsembleAccuracyPluginMetric.update(self, strategy)

    def __str__(self):
        return "Ensemble_Acc_Trained_Exp"
    
def ensemble_accuracy_metrics(
        *, 
        epoch=False,
        experience=False, 
        stream=False, 
        trained_experience=False):
    """
    Returns a list of metrics for the accuracy of the ensemble.
    :param experience: if True, the accuracy of the ensemble is computed at the
    end of each experience.
    :param stream: if True, the accuracy of the ensemble is computed at the end
    of each stream.
    :param trained_experience: if True, the accuracy of the ensemble is computed
    at the end of each experience, but only on the data seen so far.
    """
    metrics = []
    if experience:
        metrics.append(ExperienceEnsembleAccuracy())
    if stream:
        metrics.append(StreamEnsembleAccuracy())
    if trained_experience:
        metrics.append(TrainedExperienceEnsembleAccuracy())
    return metrics