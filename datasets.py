import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from torchvision import transforms, datasets

# import deeplake as dl

from avalanche.benchmarks import PermutedMNIST
from avalanche_scenarios import PACS, DomainNet

import numpy as np
import matplotlib.pyplot as plt

import os

VALID_FOLDER_DATASETS = ['CLEAR', 'SMALL_PACS', 'PACS', 'DOMAINNET']
VALID_TORCH_DATASETS = ['PERMUTEDMNIST']
VALID_AVALANCHE_DATASETS = ['PERMUTEDMNIST', 'PACS', 'DOMAINNET']
VALID_DEEPLAKE_DATASETS = ['']
VALID_DATASETS = VALID_FOLDER_DATASETS + VALID_TORCH_DATASETS + VALID_DEEPLAKE_DATASETS + VALID_AVALANCHE_DATASETS

class Dataset:
    def __init__(self, dataset_folder, dataset_name, batch_size, num_workers, shuffle, augmentation, seed):
        if dataset_name.upper() not in VALID_DATASETS:
            raise ValueError('dataset_name must be one of {}'.format(VALID_DATASETS))
        if dataset_name.upper() not in VALID_DEEPLAKE_DATASETS + VALID_AVALANCHE_DATASETS:
            if not os.path.exists(dataset_folder):
                raise Exception("Data folder does not exist")
        self.dataset_folder = dataset_folder
        self.small = False
        if dataset_name[:5].upper() == 'SMALL':
            self.small = True
            dataset_name = dataset_name[6:]
        self.dataset_name = dataset_name
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.seed = seed
        self.augmentation = augmentation

    def _create_dataset(self):
        raise NotImplementedError
    
    def _create_dataloaders(self):
        raise NotImplementedError

    def get_train_dataloaders(self):
        return self.train_dataloaders

    def get_test_dataloaders(self):
        return self.test_dataloaders

    def print_dataset_details(self, sample_image=False):
        print("Classes: ", self.dsets[0][0].classes)
        print("Class to index mapping: ", self.dsets[0][0].class_to_idx)
        print()
        for num, (train_dataset, test_dataset) in enumerate(self.dsets):
            print(f"Experience {num}:", self.experience_names[num])
            print("Train Dataset size: ", len(train_dataset))
            print("Test Dataset size: ", len(test_dataset))
            if sample_image:
                img, label = test_dataset[0]
                img = img - img.min()
                img = img / img.max()
                plt.imshow(img.permute(1, 2, 0))
                plt.show()
                print("Image shape: ", img.shape)
                print("Label: ", label, test_dataset.classes[label])
            print()
    
    def get_input_shape(self):
        return self.dsets[0][0][0][0].shape
    
    def get_mapping(self):
        return self.dsets[0][0].class_to_idx

    def get_num_classes(self):
        return len(self.dsets[0][0].classes)

class AvalancheDataset(Dataset):
    def __init__(self, dataset_folder, dataset_name, num_experiences, batch_size, num_workers, seed, order):
        super().__init__(dataset_folder, dataset_name, batch_size, num_workers, True, False, seed)
        self.num_experiences = num_experiences
        self.order = order

        if dataset_name.upper() == "PERMUTEDMNIST":
            self.scenario = PermutedMNIST(n_experiences=self.num_experiences, seed=self.seed, dataset_root=self.dataset_folder)
        elif dataset_name.upper() == "PACS":
            train_transform, test_transform = get_transforms(self.dataset_name, self.augmentation, False, seed=self.seed)
            if self.order == None:
                self.order = 'PACS'
            self.scenario = PACS(self.dataset_folder, train_transform=train_transform, test_transform=test_transform, order=self.order, seed=self.seed)
        elif dataset_name.upper() == "DOMAINNET":
            train_transform, test_transform = get_transforms(self.dataset_name, self.augmentation, False, seed=self.seed)
            if self.order == None:
                self.order = 'CIPQRS'
            self.scenario = DomainNet(self.dataset_folder, train_transform=train_transform, test_transform=test_transform, order=self.order, seed=self.seed)
        else:
            raise ValueError(f'dataset_name must be one of {VALID_AVALANCHE_DATASETS}')
        
        self.dsets, self.experience_names = self._create_dataset()
        self.train_dataloaders, self.test_dataloaders = self._create_dataloaders()

    def _create_dataset(self):
        dsets = []
        experience_names = []
        if self.dataset_name.upper() == "PERMUTEDMNIST":
            for i in range(self.num_experiences):
                train_data = self.scenario.train_stream[i].dataset
                test_data = self.scenario.test_stream[i].dataset
                dsets.append((train_data, test_data))
                experience_names.append(f"Permutation_{i+1}")
        elif self.dataset_name.upper() == "PACS" or self.dataset_name.upper() == "DOMAINNET":
            self.num_experiences = self.scenario.n_experiences
            for i in range(self.num_experiences):
                train_data = self.scenario.train_stream[i].dataset
                test_data = self.scenario.test_stream[i].dataset
                dsets.append((train_data, test_data))
                experience_names.append(self.order[i])
        return dsets, experience_names
        
    def _create_dataloaders(self):
        test_dataloaders = []
        for _, test_dset in self.dsets:
            test_loader = DataLoader(test_dset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            test_dataloaders.append(test_loader)
        return None, test_dataloaders
    
    def get_scenario(self):
        return self.scenario
    
    def print_dataset_details(self):
        print("Number of classes:", self.get_num_classes())
        print("Number of experiences:", self.num_experiences)
        for num, (train_dataset, test_dataset) in enumerate(self.dsets):
            print(f"Experience {num}:", self.experience_names[num])
            print("Train Dataset size: ", len(train_dataset))
            print("Test Dataset size: ", len(test_dataset))
            print()
    
    def get_mapping(self):
        return NotImplementedError
    
    def get_num_classes(self):
        return len(self.scenario.classes_in_experience['train'][0])

    
"""
class DeeplakeDataset(Dataset):
    def __init__(self, dataset_folder, dataset_name, batch_size, num_workers, augmentation, seed):
        super().__init__(dataset_folder, dataset_name, batch_size, num_workers, True, augmentation, seed)
        os.environ["DEEPLAKE_DOWNLOAD_PATH"] = self.dataset_folder
        self.dsets, self.experience_names = self._create_dataset()
        self.train_dataloaders, self.test_dataloaders = self._create_dataloaders()

    def print_dataset_details(self):
        raise NotImplementedError
    
    def get_input_shape(self):
        return self.dsets[0][0].images.shape[1:]
    
    def get_mapping(self):
        raise NotImplementedError
    
    def get_num_classes(self):
        raise NotImplementedError

class PermutedMNISTDataset(DeeplakeDataset):
    def __init__(self, dataset_folder, num_experiences, batch_size, num_workers, augmentation, seed):
        self.num_experiences = num_experiences
        super().__init__(dataset_folder, "PERMUTEDMNIST", batch_size, num_workers, augmentation, seed)

    def _create_dataset(self):
        train_ds = dl.load('hub://activeloop/mnist-train', access_method='local:2')
        test_ds = dl.load('hub://activeloop/mnist-test', access_method='local:2')
        dsets = [(train_ds, test_ds)]
        experience_names = [f"Permutation_{i+1}" for i in range(self.num_experiences)]
        return dsets, experience_names
    
    def _create_dataloaders(self):
        train_dataloaders = []
        test_dataloaders = []
        for train_ds, test_ds in self.dsets:
            for i in range(self.num_experiences):
                train_transform, test_transform = get_transforms(self.dataset_name, self.augmentation, self.small, seed=i)
                train_dataloader = train_ds.pytorch(num_workers=self.num_workers, batch_size=self.batch_size, 
                                                    shuffle=True, transform={'images': train_transform, 'labels': None},
                                                    decode_method={'images': 'pil'})
                test_dataloader = test_ds.pytorch(num_workers=self.num_workers, batch_size=self.batch_size, 
                                                    shuffle=False, transform={'images': test_transform, 'labels': None},
                                                    decode_method={'images': 'pil'})
                train_dataloaders.append(train_dataloader)
                test_dataloaders.append(test_dataloader)
        return train_dataloaders, test_dataloaders

    def print_dataset_details(self):
        print("Train set:", self.dsets[0][0].images.shape)
        print("Test set:", self.dsets[0][1].images.shape)
        print("Number of classes:", self.get_num_classes())
        print("Number of experiences:", self.num_experiences)

    def get_num_classes(self):
        return 10
"""

class TorchDataset(Dataset):
    def __init__(self, dataset_folder, dataset_name, num_experiences, batch_size=64, num_workers=4, shuffle=True, augmentation=False, seed=42):
        super().__init__(dataset_folder, dataset_name, batch_size, num_workers, shuffle, augmentation, seed)
        self.num_experiences = num_experiences
        self.dsets, self.experience_names = self._create_dataset()
        self.train_dataloaders, self.test_dataloaders = self._create_dataloaders()

    def _create_dataset(self):
        dsets = []
        experience_names = []
        if self.dataset_name.upper() == "PERMUTEDMNIST":
            for i in range(self.num_experiences):
                train_transform, test_transform = get_transforms(self.dataset_name, self.augmentation, self.small, seed=i)
                train_data = datasets.MNIST(self.dataset_folder, train=True, download=True, transform=train_transform)
                test_data = datasets.MNIST(self.dataset_folder, train=False, download=True, transform=test_transform)
                dsets.append((train_data, test_data))
                experience_names.append(f"Permutation_{i+1}")
            return dsets, experience_names
        else:
            raise ValueError(f'dataset_name must be one of {VALID_TORCH_DATASETS}')
        
    def _create_dataloaders(self):
        train_dataloaders = []
        test_dataloaders = []
        for train_dset, test_dset in self.dsets:
            train_loader = DataLoader(train_dset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
            test_loader = DataLoader(test_dset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            train_dataloaders.append(train_loader)
            test_dataloaders.append(test_loader)
        return train_dataloaders, test_dataloaders

class FolderDataset(Dataset):
    def __init__(self, dataset_folder, dataset_name, order=None,
                 train_pct=0.7, batch_size=64, num_workers=4, 
                 shuffle=True, augmentation=False, seed=42):
        super().__init__(dataset_folder, dataset_name, batch_size, num_workers, shuffle, augmentation, seed)
        self.order = order
        self.train_pct = train_pct
        self.dsets, self.experience_names = self._create_dataset()
        self.train_dataloaders, self.test_dataloaders = self._create_dataloaders()

    def _create_dataset(self):
        train_transform, test_transform = get_transforms(self.dataset_name, self.augmentation, self.small)
        dsets = []
        experience_names = []
        for directory in os.listdir(self.dataset_folder):
            train_data = datasets.ImageFolder(os.path.join(self.dataset_folder, directory), transform=train_transform)
            test_data = datasets.ImageFolder(os.path.join(self.dataset_folder, directory), transform=test_transform)
            dsets.append((train_data, test_data))
            experience_names.append(directory)
        if self.order is not None:
            if len(self.order) != len(dsets):
                raise ValueError("Number of experiences in order does not match number of experiences in dataset")
            dsets = [dsets[i] for i in self.order]
            experience_names = [experience_names[i] for i in self.order]
        return dsets, experience_names
    
    def _create_dataloaders(self):
        train_dataloaders = []
        test_dataloaders = []
        for train_dset, test_dset in self.dsets:
            np.random.seed(self.seed)
            num_train = len(train_dset)
            indices = list(range(num_train))
            split = int(np.floor(self.train_pct * num_train))
            if self.shuffle:
                np.random.shuffle(indices)
            train_idx, test_idx = indices[:split], indices[split:]
            train_sampler = SubsetRandomSampler(train_idx, generator=torch.Generator().manual_seed(self.seed))
            test_sampler = SubsetRandomSampler(test_idx, generator=torch.Generator().manual_seed(self.seed))
            train_loader = DataLoader(train_dset, batch_size=self.batch_size, sampler=train_sampler, num_workers=self.num_workers)
            test_loader = DataLoader(test_dset, batch_size=self.batch_size, sampler=test_sampler, num_workers=self.num_workers)
            train_dataloaders.append(train_loader)
            test_dataloaders.append(test_loader)
        return train_dataloaders, test_dataloaders

def get_transforms(dataset_name, augmentation, small, seed=None):
    if dataset_name.upper() == "PERMUTEDMNIST":
        normalize = transforms.Normalize(
            mean=[0.1307],
            std=[0.3081],
        )
        idx_permute = torch.randperm(784, generator=torch.Generator().manual_seed(seed))
        permute = transforms.Lambda(lambda x: x.view(-1)[idx_permute].view((1, 28, 28)))
        trans = transforms.Compose([transforms.ToTensor(), normalize, permute])
        return trans, trans
    
    elif dataset_name.upper() in ("CLEAR", "DOMAINNET"):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    elif dataset_name.upper() == "PACS":
        normalize = transforms.Normalize(
            mean=[0.7645, 0.7449, 0.7162],
            std=[0.3093, 0.3186, 0.3469],
        )
    else:
        raise ValueError('dataset_name must be one of {}'.format(VALID_DATASETS))
    resize_shape = 64 if small else 256
    crop_shape = 56 if small else 224
    test_preprocess =  transforms.Compose([
            transforms.Resize(resize_shape),
            transforms.CenterCrop(crop_shape),
            transforms.ToTensor(),
            normalize,
        ])
    if augmentation:
        train_preprocess = transforms.Compose([
            transforms.Resize(resize_shape),
            transforms.RandomResizedCrop(crop_shape),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_preprocess = test_preprocess
    return train_preprocess, test_preprocess