# -*- coding: utf-8 -*-
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
#import matplotlib.pyplot as plt

import random
import numpy as np
from pathlib import Path
from sklearn.mixture import GaussianMixture

#from wideresnet import WideResNet
from resnet import *
from load_noise_data import CIFAR10, CIFAR100


parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str,
help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--corruption_prob', type=float, default=0.2,
help='label noise')
parser.add_argument('--corruption_type', '-ctype', type=str, default='unif',
help='Type of corruption ("unif" or "flip" or "flip2" or "flip_smi" or "flip_nei" or "flip_adver" or "flip_smi100" or "flip_nei100").')
parser.add_argument('--num_meta', type=int, default=1000)
parser.add_argument('--epochs', default=120, type=int,
help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
help='weight decay (default: 5e-4)')
parser.add_argument('--freq', '-p', default=100, type=int,
help='print frequency (default: 10)')
parser.add_argument('--checkpoint', type=str, default='./ckpt', help="checpoint")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
parser.add_argument('--no-cuda', action='store_true', default=False,
help='disables CUDA training')
parser.set_defaults(augment=True)


#os.environ['CUD_DEVICE_ORDER'] = "1"
#ids = [1]


args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")


print()
print(args)


def build_dataset():
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    if args.augment:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
            (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    if args.dataset == 'cifar10': 
        train_data = CIFAR10(
        root='../data', train=True, corruption_prob=args.corruption_prob,
        corruption_type=args.corruption_type, transform=train_transform, download=True, seed=args.seed)
        
        test_data = CIFAR10(root='../data', train=False, transform=test_transform, download=True, seed=args.seed)

    elif args.dataset == 'cifar100':   
        train_data = CIFAR100(
        root='../data', train=True, corruption_prob=args.corruption_prob,
        corruption_type=args.corruption_type, transform=train_transform, download=True, seed=args.seed)
        
        test_data = CIFAR100(root='../data', train=False, transform=test_transform, download=True, seed=args.seed)

    train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)

    return train_data, train_loader, test_loader


def build_model():
    model = ResNet34(args.dataset == 'cifar10' and 10 or 100)
    # model = WideResNet(args.dataset == 'cifar10' and 10 or 100)
    # print('Number of model parameters: {}'.format(
    # sum([p.data.nelement() for p in model.params()])))

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    return model

train_data, train_loader, test_loader = build_dataset()
# create model
model = build_model()
model_dict = torch.load('./ckpt/model_best_%s_%s_%s.pth' %(args.dataset, args.corruption_type, args.corruption_prob))
model.load_state_dict(model_dict)

cudnn.benchmark = True

# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss(reduction='none').cuda()


class GMM_distinguish_clean_noisy(data.Dataset):
    def __init__(self, mode, model, train_data, dataset):
        self.mode = mode
        self.model = model
        self.data = np.array(train_data.train_data)
        self.labels = np.array(train_data.train_labels)

        # get train data loss
        self.model.eval()
        losses = torch.zeros(len(self.labels))

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=128, shuffle=False,
            num_workers=args.prefetch, pin_memory=True)

        with torch.no_grad():
            for batch_idx, (inputs, targets, index) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, _ = model(inputs)
                loss =criterion(outputs, targets)
                for b in range(inputs.size(0)):
                    losses[index[b]] = loss[b]
        losses = (losses - losses.min()) / (losses.max() - losses.min())

        input_loss = losses.reshape(-1, 1)

        gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss) 
        prob = prob[:,gmm.means_.argmin()]  # 把loss比较小的那列的prob值取出来，从中取干净标签

        clean_label_index = [i for i, j in enumerate(prob) if j > 0.99]

        if dataset == 'cifar10':
            img_num_list = [100] * 10
            num_classes = 10
        elif dataset == 'cifar100':
            img_num_list = [10] * 100
            num_classes = 100

        # 每个类取10个数据
        data_list_val = {}
        for j in range(num_classes):
            data_list_val[j] = [i for i, label in enumerate(self.labels) if label == j]

        idx_to_clean = []
        idx_to_noise = []

        for cls_idx, img_id_list in data_list_val.items():
            img_num = img_num_list[int(cls_idx)]
            j = 0
            for i in range(len(img_id_list)):
                if j < img_num and img_id_list[i] in clean_label_index:
                    idx_to_clean.append(img_id_list[i])
                    j += 1
                else:
                    idx_to_noise.append(img_id_list[i])

        self.clean_data = []
        self.clean_labels = []
        self.noise_data = []
        self.noise_labels = []

        self.clean_data = self.data[idx_to_clean]
        self.clean_labels = self.labels[idx_to_clean]
        self.noise_data = self.data[idx_to_noise]
        self.noise_labels = self.labels[idx_to_noise]

    def __getitem__(self, index):
        if self.mode == 'clean':
            img, target = self.clean_data[index], self.clean_labels[index]
        elif self.mode == 'noise':
            img, target = self.noise_data[index], self.noise_labels[index]
        return img, target

    def __len__(self):
        if self.mode == 'clean':
            return len(self.clean_labels)
        elif self.mode == 'noise':
            return len(self.noise_labels)

clean_data = GMM_distinguish_clean_noisy('clean', model, train_data, 'cifar10')
print('len clean data', len(clean_data))
noise_data = GMM_distinguish_clean_noisy('noise', model, train_data, 'cifar10')
print('len noise data', len(noise_data))

clean_loader = torch.utils.data.DataLoader(
    clean_data, batch_size=10, shuffle=True,
    num_workers=2, pin_memory=True)

# for batch_idx, (img, target) in enumerate(clean_loader):
#     if batch_idx == 1:
#         print(batch_idx)
#         print(img[1])
#         print(target[1])


    