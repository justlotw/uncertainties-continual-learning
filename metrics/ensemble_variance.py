from typing import List, Union, Dict, Set

import torch
from torch import Tensor

from avalanche.evaluation import Metric, PluginMetric, GenericPluginMetric

import sys 
sys.path.append('../code_extension/metrics/')  # Need to solve this problem
from vector_mean import VectorMean
from collections import defaultdict


class EnsembleVariance(Metric[Tensor]):
    def __init__(self, num_classes):
        self._mean_variance = VectorMean(num_classes)
        self.num_classes = num_classes

    @torch.no_grad()
    def update(self, predicted_y: Tensor):
        predicted_y = predicted_y.cpu()        
        softmax = torch.nn.Softmax(dim=1)
        predicted_y = softmax(predicted_y)

        if predicted_y.shape[2] != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} classes, got {predicted_y.shape[1]}"
            )
        mc_variance = predicted_y.var(dim=0) # Average over mc_samples
        variance = mc_variance.mean(axis=0)
        total_patterns = len(predicted_y)
        self._mean_variance.update(variance, total_patterns)

    def result(self) -> float:
        return self._mean_variance.result()
    
    def reset(self) -> None:
        self._mean_variance.reset()

class TaskAwareEnsembleVariance(EnsembleVariance):
    def __init__(self, num_classes):
        self._mean_variance = defaultdict(EnsembleVariance(num_classes))
        self.num_classes = num_classes

    @torch.no_grad()
    def update(self, predicted_y: Tensor, task_id: Tensor):
        if isinstance(task_id, int):
            self._mean_variance[task_id].update(predicted_y)
        elif isinstance(task_id, Tensor):
            for pred, task in zip(predicted_y, task_id):
                if isinstance(task, Tensor):
                    task = task.item()
                self._mean_variance[task].update(pred.unsqueeze(0))
        else:
            raise ValueError(
                f"Task label type: {type(task_id)}, "
                f"expected int/float or Tensor"
            )
        
    def result(self, task_label=None) -> Dict[int, float]:
        assert task_label is None or isinstance(task_label, int)
        if task_label is None:
            return {
                task_label: self._mean_variance[task_label].result()
                for task_label in self._mean_variance
            }
        else:
            return {task_label: self._mean_variance[task_label].result()}
        
    def reset(self, task_label=None) -> None:
        assert task_label is None or isinstance(task_label, int)
        if task_label is None:
            self._mean_variance = defaultdict(EnsembleVariance)
        else:
            self._mean_variance[task_label].reset()

class EnsembleVariancePluginMetric(GenericPluginMetric[float]):
    def __init__(self, num_classes, reset_at, emit_at, mode, split_by_task=False):
        self.split_by_task = split_by_task
        if self.split_by_task:
            self._variance = TaskAwareEnsembleVariance(num_classes)
        else:
            self._variance = EnsembleVariance(num_classes)
        super(EnsembleVariancePluginMetric, self).__init__(
            self._variance, reset_at=reset_at, emit_at=emit_at, mode=mode
        )

    def update(self, strategy):
        if isinstance(self._variance, EnsembleVariance):
            self._variance.update(strategy.mb_ensem_outputs)
        elif isinstance(self._variance, TaskAwareEnsembleVariance):
            self._variance.update(
                strategy.mb_ensem_outputs, strategy.mb_task_id
            )
        else:
            assert False, "should never get here."

    def result(self, strategy):
        return self._metric.result()
    
    def reset(self, strategy):
        self._metric.reset()

class ExperienceEnsembleVariance(EnsembleVariancePluginMetric):
    def __init__(self, num_classes):
        super(ExperienceEnsembleVariance, self).__init__(num_classes,
            reset_at="experience", emit_at="experience", mode="eval"
        )

    def __str__(self):
        return "Ensemble_Variance_Exp"
    
class StreamEnsembleVariance(EnsembleVariancePluginMetric):
    def __init__(self, num_classes):
        super(StreamEnsembleVariance, self).__init__(num_classes, 
            reset_at="stream", emit_at="stream", mode="eval"
        )

    def __str__(self):
        return "Ensemble_Variance_Stream"
    
class TrainedExperienceEnsembleVariance(EnsembleVariancePluginMetric):
    def __init__(self, num_classes):
        super(TrainedExperienceEnsembleVariance, self).__init__(num_classes, 
            reset_at="stream", emit_at="stream", mode="eval",
        )
        self._current_experience=0

    def after_training_exp(self, strategy) -> None:
        self._current_experience = strategy.experience.current_experience
        EnsembleVariancePluginMetric.reset(self, strategy)
        return EnsembleVariancePluginMetric.after_training_exp(self, strategy)

    def update(self, strategy):
        if strategy.experience.current_experience <= self._current_experience:
            EnsembleVariancePluginMetric.update(self, strategy)

    def __str__(self):
        return "Ensemble_Variance_Trained_Exp"
    
def ensemble_variance_metrics(
        num_classes,
        experience=False,
        stream=False,
        trained_experience=False):
    """
    Returns a list of metrics for the variance of the ensemble.
    :param experience: if True, the variance of the ensemble is computed at the
    end of each experience.
    :param stream: if True, the variance of the ensemble is computed at the end
    of each stream.
    :param trained_experience: if True, the variance of the ensemble is computed
    at the end of each experience, but only on the data seen so far.
    """
    metrics = []
    if experience:
        metrics.append(ExperienceEnsembleVariance(num_classes))
    if stream:
        metrics.append(StreamEnsembleVariance(num_classes))
    if trained_experience:
        metrics.append(TrainedExperienceEnsembleVariance(num_classes))
    return metrics