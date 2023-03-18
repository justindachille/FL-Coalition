import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import argparse
import logging
import os
import copy
from math import *
import random
import dill as pickle

import datetime
#from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import *
from vggmodel import *
from resnetcifar import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--ft_epochs', type=int, default=5, help='number of fine tuning epochs')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--alg', type=str, default='scaffold',
                            help='fl algorithms: fedavg/fedprox/scaffold/fednova/moon')
    parser.add_argument('--net_num', type=int, default=0, help="Number of nets to load")
    parser.add_argument('--train_all_layers', default=False, required=False, action='store_true', help="Trains all layers or last two")
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')

    args = parser.parse_args()
    return args

def train_net_scaffold_ft(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, train_all_layers, device="cpu"):
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)

    if train_all_layers:
        for param in net.parameters():
            param.requires_grad = True
    else:
        # Freeze all parameters
        for param in net.parameters():
            param.requires_grad = False
        # Unfreeze last two layers
        for param in net.layer3.parameters():
            param.requires_grad = True
        for param in net.layer4.parameters():
            param.requires_grad = True


    epochs_list = []
    losses = []
    valid_accuracies = []

    for epoch in range(epochs):
        epoch_loss_collector = []
        epochs_list += [epoch]
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            # optimizer.zero_grad()
            # x.requires_grad = True
            # target.requires_grad = False
            target = target.long()

            out = net(x)
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()

            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        test_acc, conf_matrix = compute_accuracy_weighted(net, test_dataloader, train_dataloader, get_confusion_matrix=True, device=device)
        logger.info('Epoch: %d Loss: %f Valid: %f' % (epoch, epoch_loss, test_acc))

        valid_accuracies += [test_acc]
        losses += [epoch_loss]
    with open(f'FineTunedNet{str(net_id)}allLayers{train_all_layers}.pickle', 'wb') as handle:
        pickle.dump((epochs_list, valid_accuracies, losses), handle, protocol=pickle.HIGHEST_PROTOCOL)

    return test_acc


if __name__ == '__main__':
    args = get_args()
    device = torch.device(args.device)
    seed = args.init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    for net in range(args.net_num):
        print(f'{args.alg}alg{net}')
        with open(f'{args.alg}alg{net}.pickle', 'rb') as handle:
            (net_id, net, train_dl_local, test_dl_local, ft_epochs, lr, optimizer, mu) = pickle.load(handle)
        
        train_net_scaffold_ft(net_id, net, train_dl_local, test_dl_local, args.ft_epochs, args.lr, args.optimizer, args.train_all_layers)
