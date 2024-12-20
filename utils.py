import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from sklearn.linear_model import LogisticRegression
from collections import Counter
from models.vgg import vgg16_bn
from models.resnet import resnet50, resnet18

import torchvision.utils as vutils
import matplotlib.pyplot as plt
from tqdm import tqdm


FEW_SIZE = 1000
def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)
    
def get_net(name, weight_path=None, num_classes=10):
    if name == 'vgg16bn':
        model = vgg16_bn(num_classes)
    elif name == 'resnet18':
        model = resnet18(num_classes)
    elif name == 'resnet50':
        model = resnet50(num_classes)
    else:
        assert 1 == 0, 'no'
    if weight_path is not None:
        model.load_state_dict(
            torch.load(weight_path, map_location='cpu')['model_states']
            )
    return model

""" checkpoint """
def load_model(model, optimizer, scheduler, path=None):
    if os.path.isfile(path):
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model_states'])
        optimizer.load_state_dict(checkpoint['optimizer_states'])
        scheduler.load_state_dict(checkpoint['scheduler_states'])
        current_epoch = checkpoint['epoch']  
    else:
        return 0
    return current_epoch

def save_model(epoch, model, optimizer, scheduler, path=None):
    assert path is not None, "PATH !!!"
    with open('{0}.pt'.format(path), 'wb') as f:
        torch.save(
                    {
                    'epoch': epoch, 
                    'model_states': model.state_dict(), 
                    'optimizer_states': optimizer.state_dict(),
                    'scheduler_states': scheduler.state_dict(),
                    }, f
            ) 

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds)) * 100

@torch.no_grad()
def evaluate(model, dl, device):
    acc = 0.
    model.to(device)
    model.eval()
    for x, labs in dl:
        x = x.to(device)
        labs = labs.to(device)
        outputs = model(x)
        acc += accuracy(outputs, labs)
    
    return acc/len(dl)


def collect_prob(data_loader, model):
    data_loader = torch.utils.data.DataLoader(
        data_loader.dataset, batch_size=1, shuffle=False
    )
    prob = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
            data, target = batch
            output = model(data)
            if isinstance(output, tuple):
                output = output[0]
            prob.append(F.softmax(output, dim=-1).data)
    return torch.cat(prob)


# https://arxiv.org/abs/2205.08096
def get_membership_attack_data(retain_loader, forget_loader, test_loader, model):
    retain_prob = collect_prob(retain_loader, model)
    forget_prob = collect_prob(forget_loader, model)
    test_prob = collect_prob(test_loader, model)

    X_r = (
        torch.cat([entropy(retain_prob), entropy(test_prob)])
        .cpu()
        .numpy()
        .reshape(-1, 1)
    )
    Y_r = np.concatenate([np.ones(len(retain_prob)), np.zeros(len(test_prob))])

    X_f = entropy(forget_prob).cpu().numpy().reshape(-1, 1)
    Y_f = np.concatenate([np.ones(len(forget_prob))])
    return X_f, Y_f, X_r, Y_r


# https://arxiv.org/abs/2205.08096
def get_membership_attack_prob(retain_loader, forget_loader, test_loader, model):
    print("MIA...")
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(
        retain_loader, forget_loader, test_loader, model
    )
    print('clf')
    clf = LogisticRegression(
        class_weight="balanced", solver="lbfgs", multi_class="multinomial"
    )
    clf.fit(X_r, Y_r)
    results = clf.predict(X_f)
    print("MIA done")
    return results.mean()


def get_class_indices(datas, classes_l, perc=None):
    
    arr_targets = np.array(datas.targets)
    
    tmp = arr_targets == classes_l[0]
    
    for c in classes_l[1:]:
        tmp += arr_targets == c 
    
    ttmp = tmp == False
    tmp_list = np.where(tmp)[0].tolist()
    ttmp_list = np.where(ttmp)[0].tolist()
    if perc is not None:
        stop_idx = int(perc * len(tmp_list))
        ttmp_list = ttmp_list + tmp_list[stop_idx:]
        tmp_list = tmp_list[:stop_idx]
    return tmp_list, ttmp_list
        
        
        
def get_ratio(dataset_original, indices):
    c_o = Counter(dataset_original.targets)
    c_f = Counter(
        dataset_original.targets[i] for i in indices
    )
    ratioes = np.empty(len(c_o))
    for k in c_o.keys():
        ratioes[k] = c_f[k]/c_o[k]
    return ratioes

def random_forget_datasets(trainset_origin, forgetset_origin, trainset_ow_aug, validset,
                              forget_perc, indices=None, few_size=FEW_SIZE):
    if indices is None:
        indices = torch.randperm(len(trainset_origin))
    forget_size = int(len(trainset_origin)*forget_perc)
    forget_train_ds = torch.utils.data.Subset(forgetset_origin, indices[:forget_size])
    forget_train_ds_one = torch.utils.data.Subset(trainset_origin, indices[:forget_size])
    forget_train_ds_wo_aug = torch.utils.data.Subset(trainset_ow_aug, indices[:forget_size])
    retain_train_ds = torch.utils.data.Subset(trainset_origin, indices[forget_size:])
    retain_train_ds_wo_aug = torch.utils.data.Subset(trainset_ow_aug, indices[forget_size:])

    forget_valid_ds = torch.utils.data.Subset(trainset_ow_aug, indices[:forget_size])
    retain_valid_ds = validset

    few_retain_ds = torch.utils.data.Subset(forgetset_origin, indices[forget_size:][:few_size])
    return (forget_train_ds, retain_train_ds, forget_valid_ds, 
            retain_valid_ds, indices, few_retain_ds, 
            forget_train_ds_wo_aug, retain_train_ds_wo_aug, forget_train_ds_one)

def class_forget_datasets(trainset_origin, forgetset_origin, trainset_ow_aug, validset,
                             classes_list:list, few_size=FEW_SIZE):
    forget_indices, retain_indices = get_class_indices(validset, classes_list)
    forget_valid_ds = torch.utils.data.Subset(validset, forget_indices)
    retain_valid_ds = torch.utils.data.Subset(validset, retain_indices)
    
    forget_indices, retain_indices = get_class_indices(trainset_origin, classes_list)
    forget_train_ds = torch.utils.data.Subset(forgetset_origin, forget_indices)
    forget_train_ds_one = torch.utils.data.Subset(trainset_origin, forget_indices)
    forget_train_ds_wo_aug = torch.utils.data.Subset(trainset_ow_aug, forget_indices)
    retain_train_ds = torch.utils.data.Subset(trainset_origin, retain_indices)
    retain_train_ds_wo_aug = torch.utils.data.Subset(trainset_ow_aug, retain_indices)

    few_retain_ds = torch.utils.data.Subset(forgetset_origin, retain_indices[:few_size])
    
    return (forget_train_ds, retain_train_ds, forget_valid_ds, 
            retain_valid_ds, forget_indices, few_retain_ds, 
            forget_train_ds_wo_aug, retain_train_ds_wo_aug, forget_train_ds_one)

def selective_forget_datasets_pre(trainset_origin, forgetset_origin, trainset_ow_aug, validset,
                             classes_list:list, perc, few_size=FEW_SIZE, indices=None):
    forget_indices, retain_indices = get_class_indices(trainset_origin, classes_list, perc)
        
    forget_train_ds = torch.utils.data.Subset(forgetset_origin, forget_indices)
    forget_train_ds_wo_aug = torch.utils.data.Subset(trainset_ow_aug, forget_indices)
    forget_train_ds_one = torch.utils.data.Subset(trainset_origin, forget_indices)
    retain_train_ds = torch.utils.data.Subset(trainset_origin, retain_indices)
    retain_train_ds_wo_aug = torch.utils.data.Subset(trainset_ow_aug, retain_indices)
    
    
    forget_valid_ds = torch.utils.data.Subset(trainset_ow_aug, forget_indices)
    retain_valid_ds = validset

    few_retain_ds = torch.utils.data.Subset(forgetset_origin, retain_indices[:few_size])
    return (forget_train_ds, retain_train_ds, forget_valid_ds,
            retain_valid_ds, forget_indices, few_retain_ds, 
            forget_train_ds_wo_aug, retain_train_ds_wo_aug, forget_train_ds_one)


def selective_forget_datasets(trainset_origin, forgetset_origin, trainset_ow_aug, validset,
                             classes_list:list, perc, few_size=FEW_SIZE, indices=None):
    forget_indices, retain_indices = get_class_indices(trainset_origin, classes_list, perc)
        
    forget_train_ds = torch.utils.data.Subset(forgetset_origin, forget_indices)
    forget_train_ds_wo_aug = torch.utils.data.Subset(trainset_ow_aug, forget_indices)
    forget_train_ds_one = torch.utils.data.Subset(trainset_origin, forget_indices)
    retain_train_ds = torch.utils.data.Subset(trainset_origin, retain_indices)
    retain_train_ds_wo_aug = torch.utils.data.Subset(trainset_ow_aug, retain_indices)
    
    forget_indices_valid, retain_indices_valid = get_class_indices(validset, classes_list)
    forget_valid_ds = torch.utils.data.Subset(validset, forget_indices_valid)
    retain_valid_ds = torch.utils.data.Subset(validset, retain_indices_valid)
    
    few_retain_ds = torch.utils.data.Subset(forgetset_origin, retain_indices[:few_size])
    return (forget_train_ds, retain_train_ds, forget_valid_ds,
            retain_valid_ds, forget_indices, few_retain_ds, 
            forget_train_ds_wo_aug, retain_train_ds_wo_aug, forget_train_ds_one)

def evaluate_metric(
    models:list,
    retain_valid_dl,
    forget_valid_dl,
    retain_train_dl,
    forget_train_dl,
    valid_dl,
    device,
    is_mia = True
):
    retain_acc = [evaluate(model, retain_train_dl, device) for model in models]
    forget_acc = [evaluate(model, forget_train_dl, device) for model in models]
    retain_tacc = [evaluate(model, retain_valid_dl, device) for model in models]
    forget_tacc = [evaluate(model, forget_valid_dl, device) for model in models]
    if is_mia: 
        mia = [get_membership_attack_prob(retain_train_dl, forget_train_dl, valid_dl, model)
               for model in models]
    else:
        mia =None
    return retain_acc, forget_acc, retain_tacc, forget_tacc, mia
                
def freeze_bnlayer(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()
            

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]



class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
    
    

def play_show(X_imgs,device, N=1, t=None):
    
    plt.figure(figsize=(6,6))
    plt.axis("off")
    plt.title(t)
    plt.imshow(np.transpose(vutils.make_grid(
        X_imgs.to(device), nrow=10, padding=2, normalize=True).cpu(), (1, 2, 0)))

    
    
    
    
    
    

