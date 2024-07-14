import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST, SVHN
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from typing import Sequence
from tqdm import tqdm
from PIL import Image


class CustomSubsetRandomSampler(SubsetRandomSampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    def __init__(self, indices: Sequence[int], generator=None) -> None:
        super().__init__(indices, generator)
    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

class CIFAR10_Preprocess(CIFAR10):
    def __init__(self, root: str, train: bool = True, transform = None, target_transform = None, download: bool = False) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ])
        self.preprocess()
        print("test")
    
    def preprocess(self):
        trainset_tensor = torch.empty(self.data.shape).transpose(1, -1)
        for idx, img in tqdm(enumerate(self.data)):
            img = Image.fromarray(img)
            trainset_tensor[idx] = self.transform(img)
        self.data = trainset_tensor
        self.targets = torch.tensor(self.targets, dtype=torch.long)
    
    def __getitem__(self, index: int):
        return self.data[index], self.targets[index]

class CIFAR100_Preprocess(CIFAR100):
    def __init__(self, root: str, train: bool = True, transform = None, target_transform = None, download: bool = False) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.preprocess()
        print("test")
    
    def preprocess(self):
        trainset_tensor = torch.empty(self.data.shape).transpose(1, -1)
        for idx, img in tqdm(enumerate(self.data)):
            img = Image.fromarray(img)
            trainset_tensor[idx] = self.transform(img)
        self.data = trainset_tensor
        self.targets = torch.tensor(self.targets, dtype=torch.long)
    
    def __getitem__(self, index: int):
        return self.data[index], self.targets[index]

class MNIST_Preprocess(MNIST):
    def __init__(self, root: str, train: bool = True, transform = None, target_transform = None, download: bool = False) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x),
            transforms.Resize(32)
            ])
        self.preprocess()
        print("test")
    
    def preprocess(self):
        trainset_tensor = torch.empty(self.data.shape[0], 3, 32, 32)
        for idx, img in tqdm(enumerate(self.data)):
            # img = Image.fromarray(img)
            img = img.unsqueeze(0)
            trainset_tensor[idx] = self.transform(img)
        self.data = trainset_tensor
        self.targets = torch.tensor(self.targets, dtype=torch.long)
    
    def __getitem__(self, index: int):
        return self.data[index], self.targets[index]

class FashionMNIST_Preprocess(FashionMNIST):
    def __init__(self, root: str, train: bool = True, transform = None, target_transform = None, download: bool = False) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x),
            transforms.Resize(32)
            ])
        self.preprocess()
        print("test")
    
    def preprocess(self):
        trainset_tensor = torch.empty(self.data.shape[0], 3, 32, 32)
        for idx, img in tqdm(enumerate(self.data)):
            # img = Image.fromarray(img)
            img = img.unsqueeze(0)
            trainset_tensor[idx] = self.transform(img)
        self.data = trainset_tensor
        self.targets = torch.tensor(self.targets.clone().detach(), dtype=torch.long)
    
    def __getitem__(self, index: int):
        return self.data[index], self.targets[index]

class SVHN_Preprocess(SVHN):
    def __init__(self, root: str, split: str, transform = None, target_transform = None, download: bool = False) -> None:
        super().__init__(root, split, transform, target_transform, download)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.preprocess()
        print("test")
    
    def preprocess(self):
        trainset_tensor = torch.empty(self.data.shape)
        for idx, img in tqdm(enumerate(self.data)):
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))
            trainset_tensor[idx] = self.transform(img)
        self.data = trainset_tensor
        self.targets = torch.tensor(self.labels, dtype=torch.long)
    
    def __getitem__(self, index: int):
        return self.data[index], self.targets[index]

def make_dataset(dataset_name, data_dir, num_data, batch_size=128, SOTA=False):
    if dataset_name == 'cifar10':
        print('Dataset: CIFAR10.')

        trainset = CIFAR10_Preprocess(root=data_dir, train=True, download=True)

        testset = CIFAR10_Preprocess(root=data_dir, train=False, download=True)
        num_classes = 10    
    elif dataset_name == "cifar100":
        print('Dataset: CIFAR100.')
        trainset = CIFAR100_Preprocess(root=data_dir, train=True, download=True)
        testset = CIFAR100_Preprocess(root=data_dir, train=False, download=True)
        num_classes = 100
    elif dataset_name == 'mnist':
        print('Dataset: MNIST.')
        trainset = MNIST_Preprocess(root=data_dir, train=True, download=True)
        testset = MNIST_Preprocess(root=data_dir, train=False, download=True)
        # trainset = MNIST(root=data_dir, train=True, download=True, transform=transforms.Compose([
        #     transforms.Grayscale(3),
        #     transforms.Resize(32),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        #     ]))

        # testset = MNIST(root=data_dir, train=False, download=True, transform=transforms.Compose([
        #     transforms.Grayscale(3),
        #     transforms.Resize(32),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        #     ]))
        num_classes = 10
    elif dataset_name == "fashionmnist":
        print('Dataset: FashionMNIST.')
        trainset = FashionMNIST_Preprocess(root=data_dir, train=True, download=True)
        testset = FashionMNIST_Preprocess(root=data_dir, train=False, download=True)
        num_classes = 10
    elif dataset_name == 'svhn':
        print('Dataset: SVHN.')
        trainset = SVHN_Preprocess(root=data_dir, split="train", download=True)
        testset = SVHN_Preprocess(root=data_dir, split="test", download=True)
        num_classes = 10
    
    print("Done normalizing dataset")
    total_sample_size = sum(num_data)
    cnt_dict = dict()
    total_cnt = 0
    indices = []
    for i in range(len(trainset)):

        if total_cnt == total_sample_size:
            break

        # label = trainset[i][1]
        label = int(trainset.targets[i])
        if label not in cnt_dict:
            cnt_dict[label] = 1
            total_cnt += 1
            indices.append(i)
        else:
            if cnt_dict[label] == num_data[label]:
                continue
            else:
                cnt_dict[label] += 1
                total_cnt += 1
                indices.append(i)

    train_indices = torch.tensor(indices)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, sampler=CustomSubsetRandomSampler(train_indices), num_workers=1,
        shuffle=False)

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)

    return trainloader, testloader, num_classes