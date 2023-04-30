import subprocess
import time
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--C_size', type=int, default=8000, help='Data points that C has')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--partition', type=str, default='noniid-labeldir', help='the data partitioning strategy')
    parser.add_argument('--coalitions', nargs='+', type=str, default=['ABC', 'BC', 'AC', 'A', 'B', 'C'])
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    runtime_settings = get_args()

    args_dict = {
        '--model': ['resnet18'],
        '--dataset': ['cifar10'],
        '--lr': [0.001],
        '--batch-size': [64],
        '--epochs': [10],
        '--ft_epochs': [10],
        '--n_parties': [3],
        '--comm_round': [50],
        '--partition': [runtime_settings.partition],
        '--C_size': [runtime_settings.C_size],
        '--datadir': ['./data/'],
        '--logdir': ['./logs/'],
        '--init_seed': [2],
        '--optimizer': ['adam'],
        '--alg': ['fedavg'],
        '--beta': [runtime_settings.beta],
        '--abc': runtime_settings.coalitions,
    }

    cmd_base = 'python scaffold_train.py'
    cmds = ['']
    for arg, values in args_dict.items():
        new_cmds = []
        for value in values:
            for cmd in cmds:
                new_cmds.append(cmd + f' {arg} {value}')
        cmds = new_cmds
    # [print(cmd, '\n') for cmd in cmds]

    # Run the commands
    for cmd in cmds:
        full_cmd = f'{cmd_base}{cmd}'
        subprocess.call(full_cmd, shell=True)
        time.sleep(1)
