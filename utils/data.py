import os
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torchtoolbox.transform import Cutout


def get_CIFAR10(root='~/data/cifar10'):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        Cutout(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if os.path.exists(root):
        print("CIFAR10 exists, preparing w/o downloading")
        train_set = datasets.CIFAR10(root=root,
                                     train=True,
                                     transform=transform_train,
                                     download=False)
        test_set = datasets.CIFAR10(root=root,
                                    train=False,
                                    transform=transform_test,
                                    download=False)

    if not os.path.exists(root):
        # download the dataset
        train_set = datasets.CIFAR10(root=root,
                                     train=True,
                                     transform=transform_train,
                                     download=True)
        test_set = datasets.CIFAR10(root=root,
                                    train=False,
                                    transform=transform_test,
                                    download=True)

    num_classes = 10

    return train_set, test_set, num_classes


def get_MNIST(root='~/data/mnist'):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),      # If the z0 of MNIST is the same size as the z0 of the CIFAR-10 dataset, then this line is needed, otherwise it's not needed
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if os.path.exists(root):
        print("MNIST exists, preparing w/o downloading")
        train_set = datasets.MNIST(root=root,
                                   train=True,
                                   transform=transform,
                                   download=False)
        test_set = datasets.MNIST(root=root,
                                    train=False,
                                    transform=transform,
                                    download=False)

    if not os.path.exists(root):
        # download the dataset
        train_set = datasets.MNIST(root=root,
                                     train=True,
                                     transform=transform,
                                     download=True)
        test_set = datasets.MNIST(root=root,
                                    train=False,
                                    transform=transform,
                                    download=True)

    num_classes = 10

    return train_set, test_set, num_classes

def get_TabularData(dataset):
    root = f'./data/{dataset}/'

    if not os.path.exists(root):
        from utils.tabular.data_preprocess import read_data
        read_data(dataset)
    X_train = np.load(f'{root}X_train.npy')
    Y_train = np.load(f'{root}Y_train.npy')
    num_classes = len(set(Y_train))
    X_test = np.load(f'{root}X_test.npy')
    Y_test = np.load(f'{root}Y_test.npy')
 
    X_train = torch.Tensor(X_train)
    X_test = torch.Tensor(X_test)
    Y_train = torch.Tensor(Y_train).type(torch.LongTensor)
    Y_test = torch.Tensor(Y_test).type(torch.LongTensor)
    train_set = TensorDataset(X_train,Y_train)
    test_set = TensorDataset(X_test,Y_test)
    return train_set, test_set, num_classes


def get_data_loader(dataset, batchsize):

    if dataset == 'CIFAR10':
        train_set, test_set, num_classes = get_CIFAR10()
    elif dataset == 'MNIST':
        train_set, test_set, num_classes = get_MNIST()
    elif dataset in ['Census','Yeast','Commercial']:
        train_set, test_set, num_classes = get_TabularData(dataset)

    else:
        raise NotImplementedError("No dataset.")
        #return None

    train_loader = DataLoader(dataset=train_set,
                              batch_size=batchsize,
                              shuffle=True,
                              num_workers=2)


    test_loader = DataLoader(dataset=test_set,
                             batch_size=batchsize,
                             shuffle=False,
                             num_workers=2)

    return train_loader, test_loader, num_classes


def get_dataset(dataset, batchsize):

    if dataset == 'CIFAR10':
        train_set, test_set, num_classes = get_CIFAR10()
    elif dataset == 'MNIST':
        train_set, test_set, num_classes = get_MNIST()
    
    elif dataset in ['Census','Yeast','Commercial']:
        train_set, test_set, num_classes = get_TabularData(dataset)

    else:
        return None

    train_loader = DataLoader(dataset=train_set,
                              batch_size=batchsize,
                              shuffle=True,
                              num_workers=2)


    test_loader = DataLoader(dataset=test_set,
                             batch_size=batchsize,
                             shuffle=False,
                             num_workers=2)

    return train_loader, test_loader, num_classes, train_set, test_set
