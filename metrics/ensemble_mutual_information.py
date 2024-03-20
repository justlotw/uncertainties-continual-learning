from typing import Dict

import torch
from torch import Tensor

from avalanche.evaluation import Metric, GenericPluginMetric
from avalanche.evaluation.metrics.mean import Mean 
from avalanche.evaluation.metrics import accuracy
from collections import defaultdict

class EnsembleMutualInformation(Metric[float]):
    def __init__(self):
        self._mean_mi = Mean()
        
    @torch.no_grad()
    def update(self, predicted_y: Tensor, true_y: Tensor):
        predicted_y = predicted_y.cpu()
        softmax = torch.nn.Softmax(dim=1)
        predicted_y = softmax(predicted_y)

        mean_predictions = predicted_y.mean(dim=0) # Average over mc_samples
        epsilon = 1e-10
        entropy = -torch.sum(mean_predictions * torch.log2(mean_predictions + epsilon), dim=1)
        mutual_information = entropy - torch.mean(torch.sum(-predicted_y * torch.log2(predicted_y + epsilon), dim=-1), dim=0)
        total_patterns = len(mean_predictions)
        self._mean_mi.update(mutual_information.mean().item(), total_patterns)

    def result(self) -> float:
        return self._mean_mi.result()
    
    def reset(self) -> None:
        self._mean_mi.reset()

class TaskAwareEnsembleMutualInformation(EnsembleMutualInformation):
    def __init__(self):
        self._mean_mi = defaultdict(EnsembleMutualInformation)

    @torch.no_grad()
    def update(self, predicted_y: Tensor, true_y: Tensor, task_id: Tensor):
        if isinstance(task_id, int):
            self._mean_mi[task_id].update(predicted_y, true_y)
        elif isinstance(task_id, Tensor):
            for pred, true, task in zip(predicted_y, true_y, task_id):
                if isinstance(task, Tensor):
                    task = task.item()
                self._mean_mi[task].update(pred.unsqueeze(0), true.unsqueeze(0))
        else:
            raise ValueError(
                f"Task label type: {type(task_id)}, "
                f"expected int/float or Tensor"
            )
        
    def result(self, task_label=None) -> Dict[int, float]:
        assert task_label is None or isinstance(task_label, int)
        if task_label is None:
            return {
                task_label: self._mean_mi[task_label].result()
                for task_label in self._mean_mi
            }
        else:
            return {task_label: self._mean_mi[task_label].result()}

    def reset(self, task_label=None) -> None:
        assert task_label is None or isinstance(task_label, int)
        if task_label is None:
            self._mean_mi = defaultdict(EnsembleMutualInformation)
        else:
            self._mean_mi[task_label].reset()

class EnsembleMutualInformationPluginMetric(GenericPluginMetric[float]):
    def __init__(self, reset_at, emit_at, mode, split_by_task=False):
        self.split_by_task = split_by_task
        if self.split_by_task:
            self._mi = TaskAwareEnsembleMutualInformation()
        else:
            self._mi = EnsembleMutualInformation()
        super().__init__(self._mi, reset_at, emit_at, mode)

    def result(self, strategy):
        return self._metric.result()
    
    def reset(self, strategy):
        self._metric.reset()

    def update(self, strategy):
        if isinstance(self._mi, EnsembleMutualInformation):
            self._mi.update(strategy.mb_ensem_outputs, strategy.mb_y)
        elif isinstance(self._mi, TaskAwareEnsembleMutualInformation):
            self._mi.update(strategy.mb_ensem_outputs, strategy.mb_y, strategy.mb_task_id)
        else:
            assert False, 'should never get here'

class ExperienceEnsembleMutualInformation(EnsembleMutualInformationPluginMetric):
    def __init__(self):
        super().__init__(reset_at='experience', emit_at='experience', mode='eval')

    def __str__(self):
        return 'Ensemble_MI_Exp'

class StreamEnsembleMutualInformation(EnsembleMutualInformationPluginMetric):
    def __init__(self):
        super().__init__(reset_at='stream', emit_at='stream', mode='eval')

    def __str__(self):
        return 'Ensemble_MI_Stream'
    
class TrainedExperienceEnsembleMutualInformation(EnsembleMutualInformationPluginMetric):
    def __init__(self):
        super().__init__(reset_at='experience', emit_at='experience', mode='train')
        self._current_experience=0

    def after_training_exp(self, strategy) -> None:
        self._current_experience = strategy.experience.current_experience
        EnsembleMutualInformationPluginMetric.reset(self, strategy)
        return EnsembleMutualInformationPluginMetric.after_training_exp(self, strategy)

    def update(self, strategy):
        if strategy.experience.current_experience <= self._current_experience:
            EnsembleMutualInformationPluginMetric.update(self, strategy)

    def __str__(self):
        return 'Ensemble_MI_Trained_Exp'

def ensemble_mutual_information_metrics(
        experience: bool = False,
        stream: bool = False,
        trained_experience: bool = False):
    
    metrics = []
    if experience:
        metrics.append(ExperienceEnsembleMutualInformation())
    if stream:
        metrics.append(StreamEnsembleMutualInformation())
    if trained_experience:
        metrics.append(TrainedExperienceEnsembleMutualInformation())
    return metrics


