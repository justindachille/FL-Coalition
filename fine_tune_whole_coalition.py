import subprocess
import time
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--C_size', type=int, default=8000, help='Data points that C has')
    parser.add_argument('--beta', type=float, default=0.1, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--partition', type=str, default='noniid-labeldir', help='the data partitioning strategy')
    parser.add_argument('--coalitions', nargs='+', type=str, default=['ABC', 'AB', 'BC', 'AC'])
    parser.add_argument('--python_ver', type=str, default=None, help='If 3, append 3 to python')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    runtime_settings = get_args()

    args_dict = {
        '--ft_epochs': [100],
        '--init_seed': [2],
        '--partition': [runtime_settings.partition],
        '--alg': ['fedavg'],
        '--C_size': [runtime_settings.C_size],
        '--abc': runtime_settings.coalitions,
        '--beta': [runtime_settings.beta],
    }

    cmd_base = 'python fine_tuning.py'
    if runtime_settings.python_ver == '3':
        cmd_base = 'python3 fine_tuning.py'
    cmds = ['']
    for arg, values in args_dict.items():
        new_cmds = []
        for value in values:
            for cmd in cmds:
                new_cmds.append(cmd + f' {arg} {value}')
        cmds = new_cmds if new_cmds else ['']
    #[print(cmd, '\n') for cmd in cmds]

    # Run the commands
    for cmd in cmds:
        full_cmd = f'{cmd_base}{cmd}'
        subprocess.call(full_cmd, shell=True)
        time.sleep(1)
