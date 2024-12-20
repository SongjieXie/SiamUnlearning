# Siamese Machine Unlearning with Knowledge Vaporization and Concentration

This repository is the Pytorch implementation of [Siamese Unlearning](https://). 

## Requirements

The codes are compatible with the packages:

- pytorch 2.2.0

- torchvision 0.17.0

- tensorboardX 2.6.2.2

- scikit-learn, numpy

To install requirements:

```setup
pip install -r requirements.txt
```

## Datasets

The code can be run on the datasets such as [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html), etc. One should download the datasets in a directory (e.g., `./data/`) and change the root parameter in `dataloader_builder.py`, e.g.,

```python
root = r'./data/'
```

## Pre-Training

One can implement `main_pretrain.py` to pre-train the original and original models on the given datasets. There are three data augmentation schemes: simple, contrastive, and cutout. The retrained models can be obtained from the three unlearning scenarios: full-class, sub-class, and random unlearning.  

To train the original model, run this command:

```shell
python main_pretrain.py --mode original --model_name resnet18 --d_name cifar10 --aug simple
```

To train the retrained VGG16-BN model on CIFAR-10 under full-class unlearning scenarios, run this command:

```shell
python main_pretrain.py --mode retrain --model_name vgg16bn --unlearn_type class --unlearn_classes 1,5,9 
```

where the parameter `unlearn_classes` specifies the unlearned classes in full-class unlearning scenarios.

To train the retrained VGG16-BN model on CIFAR-10 under sub-class unlearning scenarios, run this command:

```shell
python main_pretrain.py --mode retrain --model_name vgg16bn --d_name cifar10 --unlearn_type sub-class --unlearn_classes 1 --unlearn_perc 0.9
```

where the parameter `class` is the idx of unlearned class and `unlearn_perc` is the ratio of unlearned data within the unlearned class.

To train the retrained ResNet50 model on CIFAR-100 under random forgetting scenarios with a forgetting ratio of $10\%$, sun this command:

```shell
python main_pretrain.py --mode retrain --model_name resnet50 --d_name cifar100 --unlearn_type random --unlearn_perc 0.1
```

## Unlearning

To unlearning the original model using Siamese unlearning, run this command:

```shell
python main_unlearning.py --model_name vgg16bn --d_name cifar10 --unlearn_type class --unlearn_classes 1,5,9 --path_original ./local_base --path_retrained ./local_base_retrain  
```

where the paremeters `path_original` and `path_retrained` are the path for the original models and unlearning settings, respectively.
