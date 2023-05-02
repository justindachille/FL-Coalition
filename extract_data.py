import numpy as np
import json
import argparse
import logging
import os
import re
from math import *
import dill as pickle
from itertools import product

LOGS_PATH = './logs/'
LOGS_FT_PATH = './logs_ft/'
PARAMETER_PATTERN = r"C_size=(\d+).*abc='([^']+)'.*beta=([\d.]+)"


class Coalition:
    def __init__(self, C_size, ABC, AB_C, AC_B, A_BC, A_B_C_, beta):
        self.C_size = C_size
        self.ABC = ABC
        self.AB_C = AB_C
        self.AC_B = AC_B
        self.A_BC = A_BC
        self.A_B_C_ = A_B_C_
        self.beta = beta

    def __str__(self):
        return f"Coalition(C_size={self.C_size}, ABC={self.ABC}, AB_C={self.AB_C}, AC_B={self.AC_B}, A_BC={self.A_BC}, A_B_C_={self.A_B_C_}, beta={self.beta})"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calculate', default=False, required=False, action='store_true', help='whether to calculate price values instead of loading from pickle')
    args = parser.parse_args()
    return args

def parse_logs():
    coalitions = {}
    for file_path in os.listdir(LOGS_PATH):
        fname = os.path.join(LOGS_PATH, file_path)
        if 'DS_Store' in fname:
            continue
        if os.path.isfile(fname):
            with open(fname, "r") as f:
                print(f'reading {fname}')
                lines = f.readlines()
                second_line = lines[1]
                last_line = lines[-1]

                match = re.search(PARAMETER_PATTERN, second_line)
                C_size = None
                if match:
                    C_size = int(match.group(1))
                    abc = match.group(2)
                    beta = match.group(3)

                # extract the score value using regular expressions
                match = re.search(r"New best score: ([\d.]+)", last_line)
                if match:
                    score = float(match.group(1))
                    # print(f"C_size={C_size}, abc={abc}, score={score}")

                if C_size is None:
                    raise ValueError("Improper format of log file")
                key = (C_size, beta)
                value = [score] * 3
                print(key, value)
                if key not in coalitions:
                    # create a new coalition object with the updated property
                    if abc.upper() == "ABC":
                        coalition = Coalition(C_size, value, [], [], [], [], beta)
                    elif abc.upper() == 'AB':
                        coalition = Coalition(C_size, [], value, [], [], [], beta)
                    elif abc.upper() == 'AC':
                        coalition = Coalition(C_size, [], [], value, [], [], beta)
                    elif abc.upper() == 'BC':
                        coalition = Coalition(C_size, [], [], [], value, [], beta)
                    elif abc.upper() == 'A' or abc.upper() == 'B' or abc.upper() == 'C':
                        coalition = Coalition(C_size, [], [], [], [], value, beta)
                    else:
                        continue
                    coalitions[key] = coalition
                else:
                    # update the existing coalition object with the new property value
                    coalition = coalitions[key]
                    if abc.upper() == 'ABC':
                        coalition.ABC = (value)
                    elif abc.upper() == 'AB':
                        coalition.AB_C = (value)
                    elif abc.upper() == 'AC':
                        coalition.AC_B = (value)
                    elif abc.upper() == 'BC':
                        coalition.A_BC = (value)
                    elif abc.upper() == 'A':
                        coalition.A_B_C_ = [max(coalition.A_B_C_[0], score), coalition.A_B_C_[1], coalition.A_B_C_[2]]
                    elif abc.upper() == 'B':
                        coalition.A_B_C_ = [coalition.A_B_C_[0], max(coalition.A_B_C_[1], score), coalition.A_B_C_[2]]
                    elif abc.upper() == 'C':
                        coalition.A_B_C_ = [coalition.A_B_C_[0], coalition.A_B_C_[1], max(coalition.A_B_C_[2], score)]
                    else:
                        continue
    return coalitions

def parse_ft_logs(coalitions):
    for file_path in os.listdir(LOGS_FT_PATH):
        fname = os.path.join(LOGS_FT_PATH, file_path)
        if 'DS_Store' in fname:
            continue
        if os.path.isfile(fname):
            with open(fname, "r") as f:
                print(f'reading {fname}')
                parts = fname.split('-')
                abc = parts[0].upper()
                abc = abc.split('/')[-1]
                if parts[1] == 'custom':
                    C_size = int(parts[3])
                    beta = parts[4]
                else:
                    C_size = int(parts[2])
                    beta = parts[3]
                beta = beta.replace("1", ".1", 1)
                networks = {}
                for line in f:
                    if 'Training network' in line:
                        network = line.strip().split()[-1]
                        if network not in networks:
                            networks[network] = {'best_valid_seen': 0.0}
                    elif 'Best Valid seen' in line:
                        valid_seen = float(line.strip().split()[-1])
                        if valid_seen > networks[network]['best_valid_seen']:
                            networks[network]['best_valid_seen'] = valid_seen

                if C_size is None:
                    raise ValueError("Improper format of log file")
                key = (C_size, beta)
                values = [network['best_valid_seen'] for network in networks.values()]
                print('values:', values)
                print('before:', coalitions[key])
                coalition = coalitions[key]
                print(abc.upper())
                if abc.upper() == 'ABC':
                    coalition.ABC = [max(x) for x in zip(coalition.ABC, values)]
                elif abc.upper() == 'AB':
                    values.insert(2, 0)
                    coalition.AB_C = [max(x) for x in zip(coalition.AB_C, values)]
                elif abc.upper() == 'AC':
                    values.insert(1, 0)
                    coalition.AC_B = [max(x) for x in zip(coalition.AC_B, values)]
                elif abc.upper() == 'BC':
                    values.insert(0, 0)
                    coalition.A_BC = [max(x) for x in zip(coalition.A_BC, values)]
                else:
                    continue
                print('after:', coalitions[key])
    return coalitions

if __name__ == '__main__':
    args = get_args()
    coalitions = parse_logs()
    print('coalitions:', coalitions)
    coalitions = parse_ft_logs(coalitions)

    # coalition_str_dict = {k: str(v) for k, v in coalitions.items()}
    # print(coalition_str_dict)
    
    # return coalitions
