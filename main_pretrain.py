import os
import multiprocessing as mp
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.optim.lr_scheduler as lr_scheduler

from dataloader_builder import get_datas
from train import Trainer
from utils import class_forget_datasets, random_forget_datasets,\
    selective_forget_datasets, get_net, WarmUpLR

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='original', help='modes: original and retrain')
    parser.add_argument('--model_name', type=str, default='vgg16bn', help='model: vgg16bn resnet18 resnet50 or allcnn')
    parser.add_argument('-d', '--d_name', type=str, default='cifar10', help='dataset name')
    parser.add_argument('--lr', type=float, default=1e-3, help='learn rate')
    parser.add_argument('--maxnorm', type=float, default=10., help='The max norm of flip')
    parser.add_argument('--step', type=int, default=80, help='learn rate')
    
    # for retrain
    parser.add_argument('--unlearn_type', type=str, default='sub-class', help='class ,random, or sub-class')
    parser.add_argument('--unlearn_classes', type=str, default='1', help='class or random')
    parser.add_argument('--unlearn_perc', type=float, default=0.9, help='class or random')
    
    # only for the base model
    parser.add_argument('--aug', type=str, default='simple', help='simple, contrastive, cutout')
    parser.add_argument('--back_path', type=str, default='./local_base', help='The root of trained models')
    parser.add_argument('--device', type=str, default='cuda:0', help= 'The device')
    parser.add_argument('-e', '--epoches', type=int, default=200, help='Number of epoches') #340
    
   
    parser.add_argument('--N', type=int, default=128, help='The batch size of training data')
    parser.add_argument('--n_workers', type=int, default=int(mp.cpu_count()/1.5),
        help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    
    
    return parser.parse_args()



def main_original(config):
    """ main """
    # dataloader
    trainset_origin = get_datas(config.d_name, 32, 'train', aug=config.aug)
    validset = get_datas(config.d_name, 32, 'test', aug=config.aug)
    
    train_dl = torch.utils.data.DataLoader(trainset_origin, batch_size=config.N, shuffle=True, num_workers=config.n_workers)
    vali_dl = torch.utils.data.DataLoader(validset, batch_size=config.N, shuffle=True, num_workers=config.n_workers)

    # model and optimizer
    if config.d_name == 'cifar10':
        settings_MILESTONES = [60, 120, 160]
        settings_WARM = 2
        num_classes = 10
    elif config.d_name == 'cifar100':
        settings_MILESTONES = [60, 120, 160]
        settings_WARM = 2
        num_classes = 100
    elif config.d_name == 'tinyImagenet':
        num_classes = 200
        config.epoches = 100
        settings_MILESTONES = [30, 60, 80]
        settings_WARM = 1

    model = get_net(config.model_name, None, num_classes)
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings_MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(train_dl)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * settings_WARM)

    
    # Criterion
    criterion = nn.CrossEntropyLoss()
    
    # training
    path_to_backup = os.path.join(config.back_path + '_' + config.aug, config.d_name + '-' + config.model_name)
    writer_path = os.path.join(config.back_path + '_' + config.aug, 'log')
    print(path_to_backup)
    print(writer_path)
    trainer = Trainer(model, train_dl, vali_dl,
                      optimizer, train_scheduler, warmup_scheduler, 
                      criterion, settings_WARM,
                      config.device, config.maxnorm,
                      writer_path, path_to_backup)
    trainer.train(config.epoches)
    
    
    
def main_retrained(config):
    """ main """
    # dataloader
    trainset_origin = get_datas(config.d_name, 32, 'train', aug=config.aug)
    forgetset_origin = get_datas(config.d_name, 32, 'unlearning', aug=config.aug)
    trainset_ow_aug = get_datas(config.d_name, 32, 'train_wo_aug', aug=config.aug)
    validset = get_datas(config.d_name, 32, 'test', aug=config.aug)
    

    if config.unlearn_type == 'class':
        class_list = [int(c) for c in config.unlearn_classes.split(',')]
        print(class_list)
        (forget_train_ds, 
        retain_train_ds, 
        forget_valid_ds, 
        retain_valid_ds, 
        forget_indices,
        _, _, _, _) = class_forget_datasets(
            trainset_origin, forgetset_origin, trainset_ow_aug,
            validset, class_list
        )
    elif config.unlearn_type == 'random':
        (forget_train_ds, 
        retain_train_ds, 
        forget_valid_ds, 
        retain_valid_ds, 
        forget_indices,
        _, _, _, _) = random_forget_datasets(
            trainset_origin, forgetset_origin, trainset_ow_aug,
            validset, config.unlearn_perc
        )
    elif config.unlearn_type == 'sub-class':
        class_list = [int(c) for c in config.unlearn_classes.split(',')]
        (forget_train_ds, 
        retain_train_ds, 
        forget_valid_ds, 
        retain_valid_ds, 
        forget_indices,
        _, _, _, _) = selective_forget_datasets(
            trainset_origin, forgetset_origin, trainset_ow_aug,
            validset, class_list, config.unlearn_perc
        )
    else:
        assert 1 == 0, "no such type"
        
        
    forget_train_dl = torch.utils.data.DataLoader(forget_train_ds, batch_size=config.N, shuffle=True, num_workers=config.n_workers)
    retain_train_dl = torch.utils.data.DataLoader(retain_train_ds, batch_size=config.N, shuffle=True, num_workers=config.n_workers)
    forget_valid_dl = torch.utils.data.DataLoader(forget_valid_ds, batch_size=config.N, shuffle=False, num_workers=config.n_workers)
    retain_valid_dl = torch.utils.data.DataLoader(retain_valid_ds, batch_size=config.N, shuffle=False, num_workers=config.n_workers)
    
    # model and optimizer
    if config.d_name == 'cifar10':
        settings_MILESTONES = [60, 120, 160]
        settings_WARM = 2
        num_classes = 10
    elif config.d_name == 'cifar100':
        settings_MILESTONES = [60, 120, 160]
        settings_WARM = 2
        num_classes = 100
    elif config.d_name == 'tinyImagenet':
        num_classes = 200
        config.epoches = 100
        settings_MILESTONES = [30, 60, 80]
        settings_WARM = 1

    model = get_net(config.model_name, None, num_classes)
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings_MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(retain_train_dl)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * settings_WARM)
    
    # Criterion
    criterion = nn.CrossEntropyLoss()
    
    config.back_path = config.back_path + '_' + config.mode + '_' + config.aug
    # training
    if config.unlearn_type == 'class':
        path_to_backup = os.path.join(config.back_path, config.unlearn_type + '-' + config.unlearn_classes + '-' + config.d_name + '-' + config.model_name)
    elif config.unlearn_type == 'random':
        path_to_backup = os.path.join(config.back_path, config.unlearn_type + '-' + str(config.unlearn_perc) + '-' + config.d_name + '-' + config.model_name)
    elif config.unlearn_type == 'sub-class':
        path_to_backup = os.path.join(config.back_path, config.unlearn_type + '-' + config.unlearn_classes + '-' + str(config.unlearn_perc) + '-' + config.d_name + '-' + config.model_name)
    else:
        assert 1 == 0, "no such type"
        
    writer_path = os.path.join(config.back_path, 'log')
    
    trainer = Trainer(model, retain_train_dl, retain_valid_dl,
                      optimizer, train_scheduler, warmup_scheduler, 
                      criterion, settings_WARM,
                      config.device, config.maxnorm,
                      writer_path, path_to_backup)
    trainer.train(config.epoches)
    
    with open('{0}-setting.pt'.format(path_to_backup), 'wb') as f:
        torch.save(
                    {
                    'setting': class_list if config.unlearn_type == 'class' else forget_indices, 
                    }, f
            ) 
    
if __name__ == "__main__":
    config = parse_config()
    if config.mode == 'original':
        main_original(config)
    elif config.mode == 'retrain':
        main_retrained(config)
    else:
        assert 1 == 0, "No such mode"