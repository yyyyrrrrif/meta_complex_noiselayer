# -*- coding: utf-8 -*-
import argparse
import os
import shutil
import time
import sklearn.metrics as sm
import pandas as pd
import random
import numpy as np
from pathlib import Path

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
from torch.nn.parameter import Parameter

from resnet import *
from load_corrupted_data import CIFAR10, CIFAR100


parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str,
help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--corruption_prob', type=float, default=0.8,
help='label noise')
parser.add_argument('--corruption_type', '-ctype', type=str, default='flip_smi',
help='Type of corruption ("unif, flip_smi, flip_nei")')
parser.add_argument('--checkpoint', type=str, default='./ckpt', help="checpoint")
parser.add_argument('--num_meta', type=int, default=1000)
parser.add_argument('--epochs', default=120, type=int,
help='number of total epochs to run')
parser.add_argument('--iters', default=60000, type=int,
help='number of total iters to run')
parser.add_argument('--start-epoch', default=0, type=int,
help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
help='initial learning rate')
parser.add_argument('--meta_lr', '--meta_learning-rate', default=1e-3, type=float,
help='Meta learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
help='print frequency (default: 10)')
parser.add_argument('--layers', default=28, type=int,
help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=10, type=int,
help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0, type=float,
help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='WideResNet-28-10', type=str,
help='name of experiment')
parser.add_argument('--matrix_dir', type=str, help='dir to save estimated matrix', default='./matrix/matrix_meta')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
parser.set_defaults(augment=True)


#os.environ['CUD_DEVICE_ORDER'] = "1"
#ids = [1]
args = parser.parse_args()
use_cuda = True
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

print()
print(args)

if args.dataset == 'cifar10':
    num_classes = 10
if args.dataset == 'cifar100':
    num_classes = 100

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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def forward_loss(output, target, trans_mat):
    # l_{forward}(y, h(x)) = l_{ce}(y, h(x) @ T)
    outputs = F.softmax(output, dim=1).unsqueeze(1)
    outputs = outputs @ trans_mat.cuda()
    outputs = torch.log(outputs).squeeze(1)
    loss = F.nll_loss(outputs, target)
    return loss

# fit instance denpendent noise
class MetaMachine(MetaModule):
    def __init__(self, num_classes, bias_init):
        super(MetaMachine, self).__init__()
        self.num_classes = num_classes
        self.bias_init = bias_init

        self.layer = nn.ModuleList()
        for i in range(self.num_classes):
            self.layer.append(NoiseLayer(self.bias_init[i], 512, self.num_classes))
            # self.layer.append(MetaLinear(512, num_classes))
            # self.layer.append(nn.Sequential(MetaLinear(512, 512), nn.Dropout2d(0.5), MetaLinear(512, num_classes)))

    def forward(self, x):
        # TM: transition matric
        TM = torch.tensor([]).cuda()
        for i in range(len(self.layer)):
            TM = torch.cat((TM, self.layer[i](x)), dim=1)
        # dim=0按列计算 dim=1按行计算
        TM = F.softmax(TM.reshape(-1, num_classes, num_classes), dim=2)
        # print(TM)
        return TM

def adjust_learning_rate(optimizer, epochs):
    lr = args.lr * ((0.1 ** int(epochs >= 80)) * (0.1 ** int(epochs >= 100))) # For WRN-28-10
    #lr = args.lr * ((0.1 ** int(iters >= 20000)) * (0.1 ** int(iters >= 25000))) # For ResNet32
    # log to TensorBoard
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_C_hat(model_ce, train_meta_loader):
    print("estimating transition matrix...")
    probs = []
    targets_ = torch.tensor([]).int().cuda()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(train_meta_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, ftrs = model_ce(inputs)
            pred = F.softmax(outputs, dim=1)
            probs.extend(list(pred.data.cpu().numpy()))
            targets_ = torch.cat([targets_, torch.tensor(targets).int().cuda()], dim=0)

    probs = np.array(probs, dtype=np.float32)

    C_hat = np.zeros((num_classes, num_classes))
    for label in range(num_classes):
        indices = np.arange(len(train_meta_loader.dataset.train_labels))[
            np.isclose(np.array(targets_.cpu()), label)]
        C_hat[label] = np.mean(probs[indices], axis=0, keepdims=True)

    return C_hat.astype(np.float32)


train_loader, train_meta_loader, test_loader = build_dataset()

# create model
print('build train model:\n')
model = build_model()
model_dict = torch.load('./ckpt/model_best_%s_%s_%s.pth' %(args.dataset, args.corruption_type,args.corruption_prob))
model.load_state_dict(model_dict)
print('train model:\n', model)

print('build meta machine:\n')
X = np.array(get_C_hat(model, train_meta_loader))
trans_mat_init = torch.from_numpy(X).cuda()
bias_init = torch.log(trans_mat_init + 1e-8)
meta_machine = MetaMachine(num_classes, bias_init)
print('meta_machine:\n', meta_machine)

cudnn.benchmark = True

optimizer_a = torch.optim.SGD(model.params(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
optimizer_vnet = torch.optim.SGD(meta_machine.params(), lr=args.meta_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss().cuda()

def test(model, test_loader):
    model.eval()
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            test_loss +=criterion(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

    return accuracy


def train(train_loader, train_meta_loader, model, meta_machine, optimizer_a, optimizer_vnet, epoch):
    print('\nEpoch: %d' % epoch)

    train_loss = 0
    meta_loss = 0

    train_meta_loader_iter = iter(train_meta_loader)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        model.train()
        inputs, targets = inputs.to(device), targets.to(device)

        #========meta part=========
        meta_model = build_model().cuda()
        meta_model.load_state_dict(model.state_dict())
        outputs, outputs_meta = meta_model(inputs)

        TM = meta_machine(outputs_meta)

        l_f_meta = forward_loss(outputs, targets, TM)
        meta_model.zero_grad()
        grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
        meta_lr = args.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 100))) # For ResNet32
        meta_model.update_params(lr_inner=meta_lr, source_params=grads)
        del grads

        try:
            inputs_val, targets_val = next(train_meta_loader_iter)
        except StopIteration:
            train_meta_loader_iter = iter(train_meta_loader)
            inputs_val, targets_val = next(train_meta_loader_iter)
        inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
        y_g_hat, _ = meta_model(inputs_val)
        l_g_meta = F.cross_entropy(y_g_hat, targets_val)
        prec_meta = accuracy(y_g_hat.data, targets_val.data, topk=(1,))[0]
        meta_loss += l_g_meta.item()

        optimizer_vnet.zero_grad()
        l_g_meta.backward()
        optimizer_vnet.step()

        # ==========training part==================
        outputs, outputs_meta = model(inputs)
        TM = meta_machine(outputs_meta)

        loss = forward_loss(outputs, targets, TM)
        prec_train = accuracy(outputs.data, targets.data, topk=(1,))[0]

        optimizer_a.zero_grad()
        loss.backward()
        optimizer_a.step()
        train_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print('Epoch: [%d/%d]\t'
            'Iters: [%d/%d]\t'
            'Loss: %.4f\t'
            'MetaLoss:%.4f\t'
            'Prec@1 %.2f\t'
            'Prec_meta@1 %.2f' % (
            (epoch + 1), args.epochs, batch_idx + 1, len(train_loader.dataset)/args.batch_size, (train_loss / (batch_idx + 1)),
            (meta_loss / (batch_idx + 1)), prec_train, prec_meta))
            # if (batch_idx + 1) % 100 == 0:
            # test_acc = test(model=model, test_loader=test_loader)

def train_meta(train_meta_loader, model, optimizer, epoch):
    print('\nEpoch: %d' % epoch)
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_meta_loader):
        model.train()
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}| Acc: {:.3f} ({}/{})'.format(
                epoch, batch_idx * len(inputs), len(train_meta_loader.dataset),
                100. * batch_idx / len(train_meta_loader), (train_loss / (batch_idx + 1)),
                100. * correct / total,
                correct, total))

def main():
    checkpoint_dir = Path(args.checkpoint)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0
    test_acc = test(model=model, test_loader=test_loader)

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer_a, epoch)
        adjust_learning_rate(optimizer_vnet, epoch)
        train(train_loader, train_meta_loader, model, meta_machine, optimizer_a, optimizer_vnet, epoch)
        test_acc = test(model=model, test_loader=test_loader)
        if test_acc >= best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(),
                str(checkpoint_dir) + '/' + 'model_meta_machine_%s_%s_%s.pth' % (args.dataset, args.corruption_type, args.corruption_prob))

    print('best accuracy:', best_acc)

if __name__ == '__main__':
    main()