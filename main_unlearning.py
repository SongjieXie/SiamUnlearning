import os
import multiprocessing as mp
import random
import argparse
from tqdm import tqdm
import copy
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 

from dataloader_builder import get_datas
from utils import evaluate_metric, get_net, get_ratio,\
    class_forget_datasets, random_forget_datasets,\
    selective_forget_datasets
    
from models.unlearning_model import RandomResponse

from train import Trainer, my_unlearn


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='vgg16bn', help='model: vgg16bn resnet18 resnet50 or allcnn')
    parser.add_argument('-d', '--d_name', type=str, default='cifar10', help='dataset name')
    parser.add_argument('--lr', type=float, default=1e-4, help='learn rate')
    parser.add_argument('--unlearn_type', type=str, default='class', help='class ,random, or sub-class')
    parser.add_argument('--unlearn_classes', type=str, default='1,5,9', help='class or random')
    parser.add_argument('--unlearn_perc', type=float, default=0.9, help='class or random')
    
    # only for the base model
    parser.add_argument('--aug', type=str, default='simple', help='simple, contrastive, cutout')
    parser.add_argument('--lam', type=float, default= 1.0, help='The hyperparameter of loss')
    parser.add_argument('--path_original', type=str, default='./local_base', help='The root of trained models')
    parser.add_argument('--path_retrained', type=str, default='./local_base_retrained', help='The root of trained models')
    parser.add_argument('--back_path', type=str, default='./local_base_retrain', help='The root of trained models')
    parser.add_argument('--device', type=str, default='cuda:0', help= 'The device')
    parser.add_argument('-e', '--epoches', type=int, default=3, help='Number of epoches') #340
    
   
    parser.add_argument('--N', type=int, default=128, help='The batch size of training data')
    parser.add_argument('--n_workers', type=int, default=int(mp.cpu_count()/1.5),
        help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    
    
    return parser.parse_args()

def get_datals(d_name, aug, unlearn_type, unlearn_classes, unlearn_perc, model_name, path_retrained):

    trainset_origin = get_datas(d_name, 32, 'train', aug=aug)
    forgetset_origin = get_datas(d_name, 32, 'unlearning', aug=aug)
    trainset_ow_aug = get_datas(d_name, 32, 'train_wo_aug', aug=aug)
    validset = get_datas(d_name, 32, 'test', aug=aug)
    
        
    if unlearn_type == 'class':
        path_to_retrain = os.path.join(path_retrained + '_' + aug, unlearn_type + '-' + unlearn_classes + '-' + d_name + '-' + model_name)
    elif unlearn_type == 'random':
        path_to_retrain = os.path.join(path_retrained + '_' + aug, unlearn_type + '-' + str(unlearn_perc) + '-' + d_name + '-' + model_name)
    elif unlearn_type == 'sub-class':
        path_to_retrain = os.path.join(path_retrained + '_' + aug, unlearn_type + '-' + unlearn_classes + '-' + str(unlearn_perc) + '-' + d_name + '-' + model_name)
    else:
        assert 1 == 0, "no such type"
        
    path_to_setting = '{0}-setting.pt'.format(path_to_retrain)
    path_to_retrain = '{0}.pt'.format(path_to_retrain)
        
    if os.path.isfile(path_to_setting):
        checkpoint = torch.load(path_to_setting, map_location='cpu')
        indices = checkpoint['setting']
        print(checkpoint['setting'])
    else:
        assert 1 == 0, "No such path: {0}".format(path_to_setting)


    if unlearn_type == 'class':
        class_list = [int(c) for c in unlearn_classes.split(',')]
        print(class_list)
        (forget_train_ds, 
        retain_train_ds, 
        forget_valid_ds, 
        retain_valid_ds, 
        forget_indices, 
        few_retain_ds, 
        forget_train_ds_wo_aug, 
        retain_train_ds_wo_aug, 
        forget_train_ds_one) = class_forget_datasets(
            trainset_origin, forgetset_origin, trainset_ow_aug,
            validset, class_list
        )
    elif unlearn_type == 'random':
        (forget_train_ds, 
        retain_train_ds, 
        forget_valid_ds, 
        retain_valid_ds, 
        forget_indices, 
        few_retain_ds, 
        forget_train_ds_wo_aug, 
        retain_train_ds_wo_aug, 
        forget_train_ds_one) = random_forget_datasets(
            trainset_origin, forgetset_origin, trainset_ow_aug,
            validset, unlearn_perc
        )
    elif unlearn_type == 'sub-class':
        class_list = [int(c) for c in unlearn_classes.split(',')]
        (forget_train_ds, 
        retain_train_ds, 
        forget_valid_ds, 
        retain_valid_ds, 
        forget_indices, 
        few_retain_ds, 
        forget_train_ds_wo_aug, 
        retain_train_ds_wo_aug, 
        forget_train_ds_one) = selective_forget_datasets(
            trainset_origin, forgetset_origin, trainset_ow_aug,
            validset, class_list, unlearn_perc
        )
    else:
        assert 1 == 0, "no such type"
        
    if unlearn_type == 'random':
        forget_size = int(len(trainset_origin)*unlearn_perc)
        ratioes = get_ratio(trainset_origin, indices[:forget_size])
    else:
        ratioes = get_ratio(trainset_origin, forget_indices)
        
    return path_to_retrain, ratioes, (forget_train_ds, 
                                    retain_train_ds, 
                                    forget_valid_ds, 
                                    retain_valid_ds, 
                                    forget_indices, 
                                    few_retain_ds, 
                                    forget_train_ds_wo_aug, 
                                    retain_train_ds_wo_aug, 
                                    forget_train_ds_one,
                                    validset)
        



def siamunlearning(model_name, d_name, aug,
                   unlearn_type, unlearn_classes, unlearn_perc,
                   lam, epoches, lr, path_original, path_retrained):
    path_to_retrain, ratioes, (forget_train_ds, 
                                retain_train_ds, 
                                forget_valid_ds, 
                                retain_valid_ds, 
                                forget_indices, 
                                few_retain_ds, 
                                forget_train_ds_wo_aug, 
                                retain_train_ds_wo_aug, 
                                forget_train_ds_one, 
                                validset) = get_datals(d_name, aug, unlearn_type, 
                                                      unlearn_classes, unlearn_perc, model_name, path_retrained)
    device = 'cuda:0'
    path_to_model = os.path.join(path_original + '_' +aug, d_name + '-' + model_name)
    path_to_model = '{0}.pt'.format(path_to_model)
    print(path_to_model)
    print(path_to_retrain)
    
    
    """ Model """
    num_classes = 10 if d_name == 'cifar10' else 100
    model = get_net(model_name, path_to_model, num_classes)
    
    """ Dataloader """
    forget_train_dl = torch.utils.data.DataLoader(forget_train_ds, batch_size=128, shuffle=True, num_workers=6)
    few_retain_dl = torch.utils.data.DataLoader(few_retain_ds, batch_size=128, shuffle=True, num_workers=4)
    
    # for evaluation 
    retain_valid_dl = torch.utils.data.DataLoader(retain_valid_ds, batch_size=128, shuffle=False, num_workers=4)
    forget_valid_dl = torch.utils.data.DataLoader(forget_valid_ds, batch_size=128, shuffle=False, num_workers=4)
    retain_train_dl_wo_aug = torch.utils.data.DataLoader(retain_train_ds_wo_aug, batch_size=128, shuffle=False, num_workers=6)
    forget_train_dl_wo_aug = torch.utils.data.DataLoader(forget_train_ds_wo_aug, batch_size=128, shuffle=False, num_workers=6)
    valid_dl = torch.utils.data.DataLoader(validset, batch_size=128, shuffle=False, num_workers=6)
     

    l_re = my_unlearn(model, forget_train_dl, few_retain_dl, retain_valid_dl,forget_valid_dl,
                      ratioes, num_classes, lam, device, epoches, lr,
                      few_shot=True, path_to_backup=None, concentrate_iters=0, skip=False)
    print('The accuracy of the forgetting data for each epoch: ', l_re[0])
    print('The accuracy of the remaining data for each epoch: ', l_re[1])
    models = [model]
    models.append(
        get_net(model_name, path_to_model, num_classes)
    )
    
    ra, fa, rta, fta, mia = evaluate_metric(models, 
                                        retain_valid_dl,
                                        forget_valid_dl,
                                        retain_train_dl_wo_aug,
                                        forget_train_dl_wo_aug,
                                        valid_dl,
                                        device,
                                        is_mia = True)
    
    print("remaining acc: ", ra)
    print("forgeting acc: ", fa)
    print("remaining test acc: ", rta)
    print("forgeting test acc: ", fta)
    print("MIA: ", mia)
    
    
def main(config):
    siamunlearning(
        config.model_name,
        config.d_name,
        config.aug,
        config.unlearn_type,
        config.unlearn_classes,
        config.unlearn_perc,
        lam=config.lam,
        epoches=config.epoches,
        lr=config.lr,
        path_original= config.path_original, 
        path_retrained= config.path_retrained
    )
    
if __name__ == "__main__":
    config = parse_config()
    main(config)
    