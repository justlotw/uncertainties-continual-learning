
import os 

from torchvision import datasets

from avalanche.benchmarks.generators import ni_benchmark, nc_benchmark
from avalanche.benchmarks.utils import make_classification_dataset

def PACS(dataset_root, train_transform, test_transform, order='PACS', seed=42):
    domain = {'P': 'photo', 'A': 'art_painting', 'C': 'cartoon', 'S': 'sketch'}

    list_train_dataset = []
    list_test_dataset = []

    for d in order.upper():
        if d not in domain.keys():
            raise ValueError(f'Order must be a permutation of {list(domain.keys())}')
        train_dataset = datasets.ImageFolder(os.path.join(dataset_root, domain[d], 'train'))
        test_dataset = datasets.ImageFolder(os.path.join(dataset_root, domain[d], 'val'))
        list_train_dataset.append(make_classification_dataset(train_dataset))
        list_test_dataset.append(make_classification_dataset(test_dataset))
    
    return nc_benchmark(
        list_train_dataset, list_test_dataset, task_labels=False,
        n_experiences=len(order), shuffle=False,
        class_ids_from_zero_in_each_exp=True, 
        one_dataset_per_exp=True,
        seed=seed, 
        train_transform=train_transform, eval_transform=test_transform)


def DomainNet(dataset_root, train_transform, test_transform, order='CIPQRS', seed=42):
    domain = {'C': 'clipart', 'I': 'infograph', 'P': 'painting', 'Q': 'quickdraw', 'R': 'real', 'S': 'sketch'}

    list_train_dataset = []
    list_test_dataset = []

    for d in order.upper():
        if d not in domain.keys():
            raise ValueError(f'Order must be a permutation of {list(domain.keys())}')
        train_dataset = datasets.ImageFolder(os.path.join(dataset_root, domain[d], 'train'))
        test_dataset = datasets.ImageFolder(os.path.join(dataset_root, domain[d], 'val'))
        list_train_dataset.append(make_classification_dataset(train_dataset))
        list_test_dataset.append(make_classification_dataset(test_dataset))
    
    return nc_benchmark(
        list_train_dataset, list_test_dataset, task_labels=False,
        n_experiences=len(order), shuffle=False, 
        class_ids_from_zero_in_each_exp=True, 
        one_dataset_per_exp=True,
        seed=seed, 
        train_transform=train_transform, eval_transform=test_transform)