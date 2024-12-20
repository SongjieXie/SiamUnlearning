import torch
import torchvision.datasets as dsets
import os

import torchvision.transforms as transforms
from PIL import ImageFilter
import random
from utils import Cutout

MEAN = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4865, 0.4409),
    'tinyImagenet': (0.4802, 0.4481, 0.3975)
}

STD = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
    'tinyImagenet': (0.2302, 0.2265, 0.2262)
}

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
    

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


        
def get_aug(name, size=32):
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN[name], STD[name])
    ])
    
def get_cutout(name, size=32):
    return transforms.Compose([
        transforms.RandomCrop(size, padding=int(size/8)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN[name],STD[name]),
        Cutout(n_holes=1, length=int(size/2))
    ])
        
# def simple_transform_mnist():
#     return transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#             ])

def simple_transform_cifar(name): 
    return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  
            transforms.ToTensor(),
            transforms.Normalize(MEAN[name],STD[name])
            ])
def simple_transform_test_cifar(name):
    return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN[name],STD[name])
            ])


def simple_transform_imagenet(name):
    return transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),  
            transforms.ToTensor(),
            transforms.Normalize(MEAN[name],STD[name])
            ])


def simple_transform_test_imagenet(name):
    return transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            transforms.Normalize(MEAN[name],STD[name])
        ])
    


root = r'/home/sxieat/data/'

def get_datas(name, size, mode='unlearning', aug='cutout'):
    if aug == 'cutout':
        da = get_cutout
    elif aug == 'simple':
        da = simple_transform_cifar
    elif aug == 'contrastive':
        da = get_aug

    if name == 'cifar10':
        train_flag = False
        if mode == 'unlearning':
            tran = TwoCropsTransform(da(name))
            train_flag = True
        elif mode == 'train':
            tran = da(name)
            train_flag = True
        elif mode == 'train_wo_aug':
            tran = simple_transform_test_cifar(name)
            train_flag = True
        elif mode == 'test':
            tran = simple_transform_test_cifar(name)
        else:
            assert 1 == 0, 'No such ...'
            
        dataset = dsets.CIFAR10(root, train=train_flag, transform=tran, 
                                target_transform=None, download=False) 
    
    elif name == 'cifar100':
        train_flag = False
        if mode == 'unlearning':
            tran = TwoCropsTransform(da(name))
            train_flag = True
        elif mode == 'train':
            tran = da(name)
            train_flag = True
        elif mode == 'train_wo_aug':
            tran = simple_transform_test_cifar(name)
            train_flag = True
        elif mode == 'test':
            tran = simple_transform_test_cifar(name)
        else:
            assert 1 == 0, 'No such ...'
            
        dataset =  dsets.CIFAR100(root, train=train_flag,
                                  transform=tran, target_transform=None, download=False) 
    elif name == 'tinyImagenet':
        train_flag = False
        if mode == 'unlearning':
            tran = TwoCropsTransform(da(name, 64))
            train_flag = True
        elif mode == 'train':
            tran = da(name, 64)
            train_flag = True
        elif mode == 'train_wo_aug':
            tran = simple_transform_test_imagenet(name)
            train_flag = True
        elif mode == 'test':
            tran = simple_transform_test_imagenet(name)
        else:
            assert 1 == 0, 'No such ...'
        
        if train_flag:
            datapath = os.path.join(root+'tiny-imagenet-200', 'train') 
        else:
            datapath = os.path.join(root+'tiny-imagenet-200', 'test')
        dataset = dsets.ImageFolder(datapath, tran) 
        # dataset =  dsets.CIFAR100(root, train=train_flag,
        #                           transform=tran, target_transform=None, download=False) 
    else:
        assert 1 == 0, 'No'
    
    return dataset
            

    
    
    
    


        
