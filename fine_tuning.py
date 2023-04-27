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
from itertools import product

import datetime
#from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import *
from vggmodel import *
from resnetcifar import *


MAX_EPOCHS_BEFORE_STOPPING = 10

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--ft_epochs', type=int, default=5, help='number of fine tuning epochs')
    parser.add_argument('--partition', type=str, default='noniid-labeldir', help='the data partitioning strategy')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--alg', type=str, default='scaffold',
                            help='fl algorithms: fedavg/fedprox/scaffold/fednova/moon')
    parser.add_argument('--logdir', type=str, required=False, default="./logs_ft/", help='Log directory path')
    parser.add_argument('--net_num', type=int, default=0, help="Number of nets to load")
    parser.add_argument('--train_all_layers', default=False, required=False, action='store_true', help="Trains all layers or last two")
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--abc', type=str, default=None, help='Input as ABC, AB, AC, BC, A, B, or C')
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
    best_valid_acc = 0.0
    epochs_since_improvement = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        epochs_list += [epoch]
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            target = target.long()

            out = net(x)
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()

            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

        valid_accuracies += [test_acc]
        losses += [epoch_loss]
        if epoch % 5 == 0:
            print('Epoch: %d Loss: %f Best Valid seen: %f Valid: %f' % (epoch, epoch_loss, max(valid_accuracies), test_acc))
        logger.info('Epoch: %d Loss: %f Best Valid seen: %f Valid: %f' % (epoch, epoch_loss, max(valid_accuracies), test_acc))
        if test_acc > best_valid_acc:
            best_valid_acc = test_acc
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        # If validation accuracy hasn't improved in x epochs, stop training
        if epochs_since_improvement >= MAX_EPOCHS_BEFORE_STOPPING:
            print(f'Validation accuracy hasn\'t improved in {MAX_EPOCHS_BEFORE_STOPPING} epochs. Stopping training.')
            logger.info(f'Validation accuracy hasn\'t improved in {MAX_EPOCHS_BEFORE_STOPPING} epochs. Stopping training.')
            break
    pickle_str = f'FineTunedNet{args.partition}_{str(args.abc)}_{str(net_id)}allLayers{train_all_layers}.pickle'
    pickle_data = (epochs_list, valid_accuracies, losses)
    return max(valid_accuracies), net_id, net, pickle_str, pickle_data

if __name__ == '__main__':
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)    
    args = get_args()
    log_file_name = f'{args.abc}-{args.partition}-{datetime.datetime.now().strftime("%Y-%m-%d-%H_%M-%S")}' 
    log_path=log_file_name+'.log'
    print(args.logdir, 'path', log_path)
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',
        level=logging.DEBUG,
        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    device = torch.device(args.device)
    seed = args.init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    learning_rates = [0.001, 0.01, 0.1]
    optimizers = ['sgd']
    batch_sizes = [32, 64]
    for c in args.abc:
        best_valid_acc = 0.0
        for lr, optimizer, batch_size in product(learning_rates, optimizers, batch_sizes):
            current_params = f'lr={lr}, optimizer={optimizer}, batch_size={batch_size}'
            logger.info(f'Testing {current_params}')
            if args.partition == 'custom-quantity':
                args.beta = 0.1
            filename = f'{args.partition}_{args.alg}_{args.abc}_{c}_{args.C_size}_{args.beta}.pickle'
            if os.path.exists(filename):
                with open(filename, 'rb') as handle:
                    (net_id, net, global_model, train_dl_local, test_dl_global, current_params, lr, optimizer, batch_size) = pickle.load(handle)
                    # Ensure net is using global model parameters
                    net.load_state_dict(global_model.state_dict())
            else:
                raise ValueError(f'Inappropriate filename argument {filename}')
            best_valid_from_run, net_id, net, pickle_str, pickle_data = train_net_scaffold_ft(net_id, net, train_dl_local, test_dl_global, args.ft_epochs, args.lr, args.optimizer, args.train_all_layers)
            if best_valid_from_run > best_valid_acc:
                print(f'New best score: {best_valid_from_run} found with params {current_params}')
                logger.info(f'New best score: {best_valid_from_run} found with params {current_params}')                
                best_valid_acc = best_valid_from_run
                with open(pickle_str, 'wb') as handle:
                    pickle.dump(pickle_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
