#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

# task convex1 train_convex.py --name convex --seed 1
# task convex2 train_convex.py --name convex --seed 2
# task convex3 train_convex.py --name convex --seed 3
# task convex4 train_convex.py --name convex --seed 4
# task convex5 train_convex.py --name convex --seed 5
# task convex6 train_convex.py --name convex --seed 6
# task convex7 train_convex.py --name convex --seed 7
# task convex8 train_convex.py --name convex --seed 8
# task convex9 train_convex.py --name convex --seed 9
# task convex10 train_convex.py --name convex --seed 10
# task convex11 train_convex.py --name convex --seed 11
# task convex12 train_convex.py --name convex --seed 12

# task convexe1 train_convex.py --name convex --seed e1 --extend 0.02
# task convexe2 train_convex.py --name convex --seed e2 --extend 0.02
# task convexe3 train_convex.py --name convex --seed e3 --extend 0.02
# task convexe4 train_convex.py --name convex --seed e4 --extend 0.02
# task convexe5 train_convex.py --name convex --seed e5 --extend 0.02
# task convexe6 train_convex.py --name convex --seed e6 --extend 0.02
# task convexe7 train_convex.py --name convex --seed e7 --extend 0.02
# task convexe8 train_convex.py --name convex --seed e8 --extend 0.02
# task convexe9 train_convex.py --name convex --seed e9 --extend 0.02
# task convexe10 train_convex.py --name convex --seed e10 --extend 0.02
# task convexe11 train_convex.py --name convex --seed e11 --extend 0.02
# task convexe12 train_convex.py --name convex --seed e12 --extend 0.02

# task convexe10.01 train_convex.py --name convex --seed 0.01e1 --extend 0.01
# task convexe20.01 train_convex.py --name convex --seed 0.01e2 --extend 0.01
# task convexe30.01 train_convex.py --name convex --seed 0.01e3 --extend 0.01
# task convexe40.01 train_convex.py --name convex --seed 0.01e4 --extend 0.01
# task convexe50.01 train_convex.py --name convex --seed 0.01e5 --extend 0.01
# task convexe60.01 train_convex.py --name convex --seed 0.01e6 --extend 0.01
# task convexe70.01 train_convex.py --name convex --seed 0.01e7 --extend 0.01
# task convexe80.01 train_convex.py --name convex --seed 0.01e8 --extend 0.01
# task convexe90.01 train_convex.py --name convex --seed 0.01e9 --extend 0.01
# task convexe100.01 train_convex.py --name convex --seed 0.01e10 --extend 0.01
# task convexe110.01 train_convex.py --name convex --seed 0.01e11 --extend 0.01
# task convexe120.01 train_convex.py --name convex --seed 0.01e12 --extend 0.01

from __future__ import print_function

import argparse
import csv
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model', default="ResNet18", type=str,
                    help='model type (default: ResNet18)')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=str, help='random seed')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=300, type=int,
                    help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--is_circle', dest='circle', action='store_true', default=False)
parser.add_argument('--is_left_right', dest='left_right', action='store_true', default=False)
parser.add_argument('--left', default=0, type=float,
                    help='left (default: 0.1)')
parser.add_argument('--right', default=1, type=float,
                    help='right (default: 0.9)')
parser.add_argument('--extend', default=0, type=float,
                    help='extend the edge of dataset (default: 0.01)')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
if args.augment:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
else:
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# trainset = datasets.CIFAR10(root='../data', train=True, download=False,
#                             transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset,
#                                           batch_size=args.batch_size,
#                                           shuffle=True, num_workers=8)

# testset = datasets.CIFAR10(root='../data', train=False, download=False,
#                            transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100,
#                                          shuffle=False, num_workers=8)

trainset = datasets.CIFAR100(root='../data', train=True, download=False,
                            transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size,
                                          shuffle=True, num_workers=8)

testset = datasets.CIFAR100(root='../data', train=False, download=False,
                           transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=8)

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7' + args.name + '_'
                            + str(args.seed))
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)
else:
    print('==> Building model..')
    net = models.__dict__[args.model]()

if not os.path.isdir('results'):
    os.mkdir('results')
logname = ('results/log_' + net.__class__.__name__ + '_' + args.name + '_'
           + str(args.seed) + '.csv')

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)
    print(torch.cuda.device_count())
    cudnn.benchmark = True
    print('Using CUDA..')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                      weight_decay=args.decay)


def mixup_data(x, y, circle=False, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = -1
    while(lam < args.left or lam > args.right):
        if circle:
            lam1 = np.random.rand()
            lam2 = np.random.rand()
            lam = np.sqrt(lam1 ** 2 + lam2 ** 2) 
        else:
            lam = np.random.rand()
    lam = (lam - args.left) / (args.right - args.left)
    lam = lam * (0.5 + args.extend) + 0.5

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    dist = torch.nn.MSELoss(size_average=False)(x, mixed_x)
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, dist


def mixup_criterion(criterion, pred, y_a, y_b, lam, dist):
    loss1 = criterion(pred, y_a)
    loss2 = criterion(pred, y_b)
    p_loss =  torch.abs(loss1 - loss2) / (16 + 256 * dist)
    return lam * loss1 + (1 - lam) * loss2 - (1 - lam) * p_loss


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets_a, targets_b, lam, dist = mixup_data(inputs, targets,
                                                       args.circle, use_cuda)
        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))
        outputs = net(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam, dist)
        train_loss += loss.item() * targets.size(0)

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)

        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc = 100. * float(correct) / float(total)
    avg_loss = train_loss / total
    print('Loss: %.6f | Accuracy: %.6f%% (%d/%d)' % (avg_loss, acc, correct, total))
    return (avg_loss, acc)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)


        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    acc = 100. * float(correct) / float(total)
    avg_loss = train_loss / total
    print('Loss: %.6f | Accuracy: %.6f%% (%d/%d)' % (avg_loss, acc, correct, total))
    if epoch == start_epoch + args.epoch - 1 or acc > best_acc:
        checkpoint(acc, epoch)
    if acc > best_acc:
        best_acc = acc
    return (avg_loss, acc)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.t7' + args.name + '_'
               + str(args.seed))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    if epoch >= 200:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'train acc',
                            'test loss', 'test acc', 'best acc'])

for epoch in range(start_epoch, args.epoch):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    adjust_learning_rate(optimizer, epoch)
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, train_acc, test_loss,
                            test_acc, best_acc])
    print('best_accuracy: {}'.format(best_acc))
