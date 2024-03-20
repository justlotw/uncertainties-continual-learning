
import torch
from avalanche.evaluation import Metric

class VectorMean(Metric[torch.Tensor]):
    def __init__(self, num_classes):
        self.summed = torch.zeros(num_classes)
        self.weight = torch.zeros(num_classes)
        self.num_classes = num_classes

    def update(self, value, weight=1.0):
        value = torch.tensor(value)
        weight = torch.tensor(weight)
        self.summed += value * weight
        self.weight += weight

    def result(self):
        if self.weight.sum() == 0:
            return torch.zeros(self.num_classes)
        return self.summed / self.weight
    
    def reset(self):
        self.summed = torch.zeros(self.num_classes)
        self.weight = torch.zeros(self.num_classes)

    def __add__(self, other):
        res = VectorMean(self.num_classes)
        res.summed = self.summed + other.summed
        res.weight = self.weight + other.weight
        return res
    