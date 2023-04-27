import subprocess
import time
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--C_size', type=int, default=8000, help='Data points that C has')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    runtime_settings = get_args()
    # Define the possible values for each command line argument
    args = [
        {"arg": "--model", "values": ["resnet18"]},
        {"arg": "--dataset", "values": ["cifar10"]},
        {"arg": "--lr", "values": [0.001]},
        {"arg": "--batch-size", "values": [64]},
        {"arg": "--epochs", "values": [10]},
        {"arg": "--ft_epochs", "values": [10]},
        {"arg": "--n_parties", "values": [3]},
        {"arg": "--comm_round", "values": [50]},
        {"arg": "--partition", "values": ["custom-quantity"]},
        {"arg": "--C_size", "values": [runtime_settings.C_size]},
        {"arg": "--datadir", "values": ["./data/"]},
        {"arg": "--logdir", "values": ["./logs/"]},
        {"arg": "--init_seed", "values": [2]},
        {"arg": "--optimizer", "values": ["adam"]},
        {"arg": "--alg", "values": ["fedavg"]},
        {"arg": "--beta", "values": [runtime_settings.beta]},
        {"arg": "--abc", "values": ["ABC", "BC", "AC", "A", "B", "C"]}
    ]

    # Generate all possible combinations of command line arguments
    cmds = [""]
    for arg in args:
        new_cmds = []
        for value in arg["values"]:
            for cmd in cmds:
                new_cmds.append(cmd + f" {arg['arg']} {value}")
        cmds = new_cmds
    # [print(cmd, '\n') for cmd in cmds]

    # Run the commands
    for cmd in cmds:
        subprocess.call(f"python scaffold_train.py{cmd}", shell=True)
        time.sleep(1)
