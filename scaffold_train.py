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
import sys
import copy
from math import *
import random
from itertools import product
import dill as pickle

import datetime
#from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import *
from vggmodel import *
from resnetcifar import *

MAX_EPOCHS_BEFORE_STOPPING = 10

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MLP', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='noniid-labeldir', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--ft_epochs', type=int, default=5, help='number of fine tuning epochs')
    parser.add_argument('--n_parties', type=int, default=2,  help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='scaffold',
                            help='fl algorithms: fedavg/fedprox/scaffold/fednova/moon')
    parser.add_argument('--use_projection_head', type=bool, default=False, help='whether add an additional header to model or not (see MOON)')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--loss', type=str, default='contrastive', help='for moon')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox')
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='level', help='Different level of noise or different space of noise')
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=1, help='Sample ratio for each communication round')
    parser.add_argument('--abc', type=str, default=None, help='Input as ABC, AB, AC, BC, A, B, or C')
    args = parser.parse_args()
    return args


def init_nets(net_configs, dropout_p, n_parties, args):
    nets = {net_i: None for net_i in range(n_parties)}

    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist'}:
        n_classes = 10
    for net_i in range(n_parties):
        if args.model == "vgg":
            net = vgg11()
        elif args.model == "simple-cnn":
            if args.dataset in ("cifar10", "cinic10", "svhn"):
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
            elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
            elif args.dataset == 'celeba':
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2)
        elif args.model == "vgg-9":
            if args.dataset in ("mnist", 'femnist'):
                net = ModerateCNNMNIST()
            elif args.dataset in ("cifar10", "cinic10", "svhn"):
                # logger.info("in moderate cnn")
                net = ModerateCNN()
            elif args.dataset == 'celeba':
                net = ModerateCNN(output_dim=2)
        elif args.model == "resnet":
            logger.info('using complex net')
            net = ResNet50_cifar10(num_classes=10)
        elif args.model == "resnet18":
            logger.info('using simple net')
            net = ResNet18_cifar10(num_classes=10)
        elif args.model == "vgg16":
            net = vgg16()
        else:
            logger.info("not supported yet")
            exit(1)
        nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)
    return nets, model_meta_data, layer_type


def train_net_scaffold(net_id, net, global_model, c_local, c_global, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))
    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))
    loss_total = 0.0
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    c_global_para = c_global.state_dict()
    c_local_para = c_local.state_dict()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)
                loss_total += loss.item()

                loss.backward()
                optimizer.step()

                net_para = net.state_dict()
                for key in net_para:
                    net_para[key] = net_para[key] - args.lr * (c_global_para[key] - c_local_para[key])
                net.load_state_dict(net_para)

                cnt += 1
                epoch_loss_collector.append(loss.item())


        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    c_new_para = c_local.state_dict()
    c_delta_para = copy.deepcopy(c_local.state_dict())
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    for key in net_para:
        c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (cnt * args.lr)
        c_delta_para[key] = c_new_para[key] - c_local_para[key]
    c_local.load_state_dict(c_new_para)


    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)


    logger.info(' ** Training complete **')
    return train_acc, test_acc, c_delta_para, loss_total


def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    #writer = SummaryWriter()
    loss_total = 0
    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)
                loss_total += loss.item()

                loss.backward()
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        #train_acc = compute_accuracy(net, train_dataloader, device=device)
        #test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

        #writer.add_scalar('Accuracy/train', train_acc, epoch)
        #writer.add_scalar('Accuracy/test', test_acc, epoch)

        # if epoch % 10 == 0:
        #     logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
        #     train_acc = compute_accuracy(net, train_dataloader, device=device)
        #     test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        #
        #     logger.info('>> Training accuracy: %f' % train_acc)
        #     logger.info('>> Test accuracy: %f' % test_acc)

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)


    logger.info(' ** Training complete **')
    return train_acc, test_acc, loss_total


def local_train_net(nets, selected, args, net_dataidx_map, test_dl = None, device="cpu"):
    avg_acc = 0.0
    loss_total = 0

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        trainacc, testacc, local_net_loss = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, device=device)
        loss_total += local_net_loss
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        # saving the trained models here
        # save_model(net, net_id, args)
        # else:
        #     load_model(net, net_id, device=device)
    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list, loss_total


def local_train_net_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map, test_dl = None, device="cpu"):
    avg_acc = 0.0
    loss_total = 0.0

    total_delta = copy.deepcopy(global_model.state_dict())
    for key in total_delta:
        total_delta[key] = 0.0
    c_global.to(device)
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        c_nets[net_id].to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs


        trainacc, testacc, c_delta_para, local_loss = train_net_scaffold(net_id, net, global_model, c_nets[net_id], c_global, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, device=device)
        loss_total += local_loss
        c_nets[net_id].to('cpu')
        for key in total_delta:
            total_delta[key] += c_delta_para[key]

        print("net %d final test acc %f" % (net_id, testacc))
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
    for key in total_delta:
        total_delta[key] /= args.n_parties
    c_global_para = c_global.state_dict()
    for key in c_global_para:
        if c_global_para[key].type() == 'torch.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.LongTensor)
        elif c_global_para[key].type() == 'torch.cuda.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
        else:
            #logger.info(c_global_para[key].type())
            c_global_para[key] += total_delta[key]
    c_global.load_state_dict(c_global_para)

    avg_acc /= len(selected)

    nets_list = list(nets.values())
    return nets_list, loss_total

#[4000, 12000, 33000]
def partition_data(dataset, datadir, logdir, partition, n_parties, clients_split=[3000, 3000, 3000], beta=0.4):
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    n_train = y_train.shape[0]
    if partition == "noniid-labeldir":
        clients_split = [3000, 3000, 3000]
        min_size = 0
        min_require_size = 10
        K = 10
        N = y_train.shape[0]
        net_dataidx_map = {}
        n_subsamples = sum(clients_split)
        samples_per_label = n_subsamples // K

        X_train_subsampled = []
        y_train_subsampled = []

        for k in range(K):
            indices_k = np.where(y_train == k)[0]
            indices_k_sampled = np.random.choice(indices_k, samples_per_label, replace=False)
            X_train_subsampled.append(X_train[indices_k_sampled])
            y_train_subsampled.append(y_train[indices_k_sampled])

        X_train = np.concatenate(X_train_subsampled, axis=0)
        y_train = np.concatenate(y_train_subsampled, axis=0)

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                # print(f'count of k: {k} is {np.count_nonzero(y_train == k)}')
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
            # print(f'party {j} in map len {len(net_dataidx_map[j])}')
            # print('net', net_dataidx_map.values())

    elif partition == "custom-quantity":
        clients_split = [1000, 3000, 8000]
        min_size = 0
        K = 10
        N = y_train.shape[0]
        net_dataidx_map = {}

        idxs = np.random.permutation(n_train)
        start_idx = 0
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            subsample = [x//10 for x in clients_split]
            start_idx = 0
            for i in range(n_parties):
                if i not in net_dataidx_map:
                    net_dataidx_map[i] = idx_k[start_idx:start_idx+subsample[i]]
                else:
                    net_dataidx_map[i] = np.append(net_dataidx_map[i], idx_k[start_idx:start_idx+subsample[i]])
                start_idx += subsample[i]
        for i in range(n_parties):
            print('party i:', i, 'length:', len(net_dataidx_map[i]))
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)


def train_single(net_id, net, train_dataloader, test_dataloader, arg_optimizer, arg_lr, device="cpu"):
    if arg_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=arg_lr, weight_decay=args.reg)
    elif arg_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=arg_lr, weight_decay=args.reg,
                            amsgrad=True)
    elif arg_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=arg_lr, momentum=args.rho, weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    epochs_list = []
    losses = []
    valid_accuracies = []

    for epoch in range(args.epochs):
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
        logger.info('Epoch: %d Loss: %f Best Valid seen: %f Valid: %f' % (epoch, epoch_loss, max(valid_accuracies), test_acc))
    return max(valid_accuracies), net_id, net


if __name__ == '__main__':
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    device = torch.device(args.device)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if args.abc is None:
        raise ValueError('No setup specified: choose ABC, AB, AC, BC, A, B, C')
    if args.log_file_name is None:
        args.log_file_name = f'{args.abc}-{args.partition}-{datetime.datetime.now().strftime("%Y-%m-%d-%H_%M-%S")}' 
    log_path=f'{args.log_file_name}.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',
        level=logging.DEBUG,
        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)
    logger.info(f'args: {str(args)}')
    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)
    n_classes = len(np.unique(y_train))

    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                        args.datadir,
                                                                                        args.batch_size,
                                                                                        32)

    # print("len train_dl_global:", len(train_ds_global))

    data_size = len(test_ds_global)

    # test_dl = data.DataLoader(dataset=test_ds_global, batch_size=32, shuffle=False)

    train_all_in_list = []
    test_all_in_list = []
    if args.noise > 0:
        for party_id in range(args.n_parties):
            dataidxs = net_dataidx_map[party_id]

            noise_level = args.noise
            if party_id == args.n_parties - 1:
                noise_level = 0

            if args.noise_type == 'space':
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, party_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * party_id
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
            train_all_in_list.append(train_ds_local)
            test_all_in_list.append(test_ds_local)
        train_all_in_ds = data.ConcatDataset(train_all_in_list)
        train_dl_global = data.DataLoader(dataset=train_all_in_ds, batch_size=args.batch_size, shuffle=True)
        test_all_in_ds = data.ConcatDataset(test_all_in_list)
        test_dl_global = data.DataLoader(dataset=test_all_in_ds, batch_size=32, shuffle=False)


    best_valid_acc = 0.0
    learning_rates = [0.001, 0.01]
    optimizers = ['sgd']
    batch_sizes = [64, 128]
    communication_round = []
    training_loss = []
    valid_accuracy = []

    if args.alg == 'fedavg':
        for lr, optimizer, batch_size in product(learning_rates, optimizers, batch_sizes):
            current_params = f'lr={lr}, optimizer={optimizer}, batch_size={batch_size}'
            logger.info(f'Testing {current_params}')
            # Do grid search here
            communication_round = []
            training_loss = []
            valid_accuracy = []
            logger.info('Initializing nets')
            nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
            global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
            global_model = global_models[0]

            global_para = global_model.state_dict()
            for round in range(args.comm_round):
                communication_round += [round]

                logger.info("in comm round:" + str(round))

                arr = np.arange(args.n_parties)
                selected = arr[:int(args.n_parties * args.sample)]

                net_setup = args.abc.lower()

                if net_setup == 'a':
                    selected = selected[0]
                elif net_setup == 'b':
                    selected = selected[1]
                elif net_setup == 'c':
                    selected = selected[2]
                elif net_setup == 'ab':
                    selected = selected[0:2]
                elif net_setup == 'bc':
                    selected = selected[1:3]
                elif net_setup == 'ac':
                    selected = [selected[0], selected[2]]
                if len(args.abc) == 1:
                    # Single client
                    net_id = selected
                    dataidxs = net_dataidx_map[net_id]
                    if args.noise_type == 'space':
                        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, batch_size, 32, dataidxs, noise_level, args.n_parties-1)
                    else:
                        noise_level = args.noise / (args.n_parties - 1) * net_id
                        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, batch_size, 32, dataidxs, noise_level)
                    train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, batch_size, 32)
                    best_valid_from_run, net_id, net = train_single(net_id, nets[net_id], train_dl_local, test_dl_global, optimizer, lr, device="cpu")
                    print(f'Done training, best score: {best_valid_from_run} found with params {current_params}')
                    if best_valid_from_run > best_valid_acc:
                        print(f'New best score: {best_valid_from_run} found with params {current_params}')
                        logger.info(f'New best score: {best_valid_from_run} found with params {current_params}')                
                        best_valid_acc = best_valid_from_run
                        for net_id, net in nets.items():
                            dataidxs = net_dataidx_map[net_id]

                            if args.noise_type == 'space':
                                train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
                            else:
                                noise_level = args.noise / (args.n_parties - 1) * net_id
                                train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, batch_size, 32, dataidxs, noise_level)
                            train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, batch_size, 32)
                            int_to_str = {0: 'a', 1: 'b', 2: 'c'}
                            with open(f'{args.abc}_{int_to_str[net_id]}.pickle', 'wb') as handle:
                                pickle.dump((net_id, net, train_dl_local, test_dl_global, current_params, lr, optimizer, batch_size), handle, protocol=pickle.HIGHEST_PROTOCOL)
                if len(args.abc) == 1:
                    break
                global_para = global_model.state_dict()
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

                global_para = global_model.state_dict()
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

                _, loss_total = local_train_net(nets, selected, args, net_dataidx_map, test_dl = test_dl_global, device=device)
                # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

                # update global model
                total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
                fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

                for idx in range(len(selected)):
                    net_para = nets[selected[idx]].cpu().state_dict()
                    if idx == 0:
                        for key in net_para:
                            global_para[key] = net_para[key] * fed_avg_freqs[idx]
                    else:
                        for key in net_para:
                            global_para[key] += net_para[key] * fed_avg_freqs[idx]
                global_model.load_state_dict(global_para)

                logger.info('global n_training: %d' % len(train_dl_global))
                logger.info('global n_test: %d' % len(test_dl_global))

                global_model.to(device)
                train_acc = compute_accuracy(global_model, train_dl_global, device=device)
                test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)

                logger.info('>> Global Model Train accuracy: %f' % train_acc)
                logger.info('>> Global Model Test accuracy: %f' % test_acc)
                valid_accuracy += [test_acc]
                training_loss += [loss_total]
                logger.info(' -- comm_round' + ' '.join(map(str, communication_round)) + ': valid : ' + ' '.join(map(str, valid_accuracy)))
                logger.info('best valid so far' + str(max(valid_accuracy)) + ' -- comm_round' + ' '.join(map(str, communication_round)) + ': valid : ' + ' '.join(map(str, valid_accuracy)) + ': loss : ' + ' '.join(map(str, training_loss)))
            if len(args.abc) == 1:
                continue
            if max(valid_accuracy) > best_valid_acc:
                best_valid_acc = max(valid_accuracy)
                for net_id, net in nets.items():
                    dataidxs = net_dataidx_map[net_id]

                    if args.noise_type == 'space':
                        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
                    else:
                        noise_level = args.noise / (args.n_parties - 1) * net_id
                        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
                    train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
                    if net_id in selected:
                        int_to_str = {0: 'a', 1: 'b', 2: 'c'}
                        with open(f'{args.partition}_{args.alg}_{args.abc}_{int_to_str[net_id]}.pickle', 'wb') as handle:
                            pickle.dump((net_id, net, train_dl_local, test_dl_global, current_params, lr, optimizer, batch_size), handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
                with open(f'{args.partition}_{args.alg}_{args.abc}.pickle', 'wb') as handle:
                    pickle.dump((communication_round, valid_accuracy, training_loss), handle, protocol=pickle.HIGHEST_PROTOCOL)
    elif args.alg == 'scaffold':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        c_nets, _, _ = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        c_globals, _, _ = init_nets(args.net_config, 0, 1, args)
        c_global = c_globals[0]
        c_global_para = c_global.state_dict()
        for net_id, net in c_nets.items():
            net.load_state_dict(c_global_para)

        global_para = global_model.state_dict()

        for round in range(args.comm_round):
            communication_round += [round]

            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            selected = arr[:int(args.n_parties * args.sample)]

            net_setup = args.abc.lower()

            if net_setup == 'a':
                selected = selected[0]
            elif net_setup == 'b':
                selected = selected[1]
            elif net_setup == 'c':
                selected = selected[2]
            elif net_setup == 'ab':
                selected = selected[0:2]
            elif net_setup == 'bc':
                selected = selected[1:3]
            elif net_setup == 'ac':
                selected = [selected[0], selected[2]]
            if len(args.abc) == 1:
                # Single client
                net_id = selected
                dataidxs = net_dataidx_map[net_id]
                if args.noise_type == 'space':
                    train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, args.n_parties-1)
                else:
                    noise_level = args.noise / (args.n_parties - 1) * net_id
                    train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
                train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
                train_single(net_id, nets[net_id], train_dl_local, test_dl_global, device="cpu")
                break
            global_para = global_model.state_dict()
            for idx in selected:
                nets[idx].load_state_dict(global_para)

            _, loss_total = local_train_net_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map, test_dl = test_dl_global, device=device)
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)
            valid_accuracy += [test_acc]
            training_loss += [loss_total]

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info(' -- comm_round' + ' '.join(map(str, communication_round)) + ': valid : ' + ' '.join(map(str, valid_accuracy)) + ': loss : ' + ' '.join(map(str, training_loss)))
            if test_acc > best_valid_acc:
                best_valid_acc = test_acc
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            # If validation accuracy hasn't improved in 5 epochs, stop training
            if epochs_since_improvement >= 5:
                logger.info("Validation accuracy hasn't improved in 5 epochs. Stopping training.")
                break
        for net_id, net in nets.items():
            dataidxs = net_dataidx_map[net_id]

            if args.noise_type == 'space':
                train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
            train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
            if net_id in selected:
                int_to_str = {0: 'a', 1: 'b', 2: 'c'}
                with open(f'{args.abc}_{int_to_str[net_id]}.pickle', 'wb') as handle:
                    pickle.dump((net_id, net, train_dl_local, test_dl_local, args.ft_epochs, args.lr, args.optimizer, args.mu), handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            # train_net_fedprox_ft(net_id, net, train_dl_local, test_dl_global, args.ft_epochs, args.lr, args.optimizer, args.mu)

        with open(f'{args.alg}beta{args.beta}.pickle', 'wb') as handle:
            pickle.dump((communication_round, valid_accuracy, training_loss), handle, protocol=pickle.HIGHEST_PROTOCOL)

