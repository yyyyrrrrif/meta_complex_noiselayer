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
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
#import matplotlib.pyplot as plt
import random
import numpy as np
from pathlib import Path


#from wideresnet import WideResNet
from resnet import *
from load_corrupted_data import CIFAR10, CIFAR100


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
parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')
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
        train_data_meta = CIFAR10(
        root='../data', train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
        corruption_type=args.corruption_type, transform=train_transform, download=True)
        
        train_data = CIFAR10(
        root='../data', train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
        corruption_type=args.corruption_type, transform=train_transform, download=True, seed=args.seed)
        
        test_data = CIFAR10(root='../data', train=False, transform=test_transform, download=True)

    elif args.dataset == 'cifar100':
        train_data_meta = CIFAR100(
        root='../data', train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
        corruption_type=args.corruption_type, transform=train_transform, download=True)
        
        train_data = CIFAR100(
        root='../data', train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
        corruption_type=args.corruption_type, transform=train_transform, download=True, seed=args.seed)
        
        test_data = CIFAR100(root='../data', train=False, transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)

    train_meta_loader = torch.utils.data.DataLoader(
    train_data_meta, batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)

    return train_loader, train_meta_loader, test_loader



def build_model():
    model = ResNet34(args.dataset == 'cifar10' and 10 or 100)
    # model = WideResNet(args.dataset == 'cifar10' and 10 or 100)

    # print('Number of model parameters: {}'.format(
    # sum([p.data.nelement() for p in model.params()])))

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True


    return model



def adjust_learning_rate(optimizer, epochs):
    lr = args.lr * ((0.1 ** int(epochs >= 80)) * (0.1 ** int(epochs >= 100))) # For WRN-28-10
    #lr = args.lr * ((0.1 ** int(iters >= 20000)) * (0.1 ** int(iters >= 25000))) # For ResNet32
    # log to TensorBoard
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


#=======================================================================================================
#========================================================================================================
train_loader, train_meta_loader,test_loader = build_dataset()
# create model
model = build_model()
optimizer = torch.optim.SGD(model.params(), args.lr,
momentum=args.momentum,
weight_decay=args.weight_decay)

cudnn.benchmark = True

# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss().cuda()

def train(train_loader, model, optimizer, criterion,epoch):
    print('\nEpoch: %d' % epoch)
    train_loss = 0
    correct = 0
    total = 0
    # best_acc = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        model.train()
        inputs, targets = inputs.to(device), targets.to(device)
        # print(inputs.shape)
        # print(targets.shape)
        optimizer.zero_grad()
        outputs = model(inputs)
        # print(outputs.shape)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()


        if (batch_idx + 1) % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}| Acc: {:.3f} ({}/{})'.format(
            epoch, batch_idx * len(inputs), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), (train_loss / (batch_idx + 1)), 100. * correct / total,
            correct, total))


def test(model, test_loader):
    model.eval()
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss +=criterion(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

    return accuracy



def main():
    best_ce_acc = 0
    best_acc = 0

    checkpoint_dir = Path(args.checkpoint)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer,epoch)
        train(train_loader=train_loader, model=model, optimizer=optimizer,criterion=criterion,epoch=epoch)
        val_acc = test(model=model, test_loader=train_meta_loader)
        test_acc = test(model=model, test_loader=test_loader)
        ## 正常情况
        if test_acc >= best_acc:
            best_acc = test_acc

        if val_acc >= best_ce_acc:
            best_ce_acc = val_acc
            torch.save(model.state_dict(),
                        str(checkpoint_dir) + '/' + 'model_best_%s_%s_%s.pth' % (args.dataset, args.corruption_type, args.corruption_prob))
    print('best accuracy:', best_acc)


if __name__ == '__main__':
    main()

