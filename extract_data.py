import numpy as np
import json
import argparse
import logging
import os
import re
from math import *
import dill as pickle
from itertools import product

from bestresponse import createTableFromCoalition, Coalition

LOGS_PATH = './logs/'
LOGS_FT_PATH = './logs_ft/'
PARAMETER_PATTERN = r"(?=.*C_size=(\d+))(?=.*abc='([^']+)')(?=.*beta=([\d.]+))"


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
                print(f'Reading: {fname}')
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
                score = None
                if match:
                    score = float(match.group(1))
                    # print(f"C_size={C_size}, abc={abc}, score={score}")

                if C_size is None or score is None:
                    raise ValueError("Improper format of log file")
                key = (C_size, beta)
                value = [score] * 3
                # print(key, value)
                if key not in coalitions:
                    # create a new coalition object with the updated property
                    if abc.upper() == "ABC":
                        coalition = Coalition(C_size, value, [], [], [], [0]*3, beta)
                    elif abc.upper() == 'AB':
                        coalition = Coalition(C_size, [], value, [], [], [0]*3, beta)
                    elif abc.upper() == 'AC':
                        coalition = Coalition(C_size, [], [], value, [], [0]*3, beta)
                    elif abc.upper() == 'BC':
                        coalition = Coalition(C_size, [], [], [], value, [0]*3, beta)
                    elif abc.upper() == 'A':
                        # In these single client cases, set values that aren't read
                        # to be 0, so they are overriden later with max() function
                        value[1] = 0
                        value[2] = 0
                        coalition = Coalition(C_size, [], [], [], [], value, beta)
                    elif abc.upper() == 'B':
                        value[0] = 0
                        value[2] = 0
                        coalition = Coalition(C_size, [], [], [], [], value, beta)
                    elif abc.upper() == 'C':
                        value[0] = 0
                        value[1] = 0
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
                        print('Invalid input')
                        continue
    return coalitions

def parse_ft_logs(coalitions):
    for file_path in os.listdir(LOGS_FT_PATH):
        fname = os.path.join(LOGS_FT_PATH, file_path)
        if 'DS_Store' in fname:
            continue
        if os.path.isfile(fname):
            with open(fname, "r") as f:
                print(f'Reading: {fname}')
                parts = fname.split('-')
                abc = parts[0].upper()
                abc = abc.split('/')[-1]
                C_size = int(parts[3])
                beta = parts[4].replace("1", ".1", 1)
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
                # if C_size == 10000:
                    # print('Networks:', networks)
                    # print('values:', values)
                #     print('before:', coalitions[key])
                coalition = coalitions[key]
                print('values:', networks)
                if abc.upper() == 'ABC':
                    if '0' in networks:
                        coalition.ABC[0] = max(coalition.ABC[0], networks['0']['best_valid_seen'])
                    if '1' in networks:
                        coalition.ABC[1] = max(coalition.ABC[1], networks['1']['best_valid_seen'])
                    if '2' in networks:
                        coalition.ABC[2] = max(coalition.ABC[2], networks['2']['best_valid_seen'])
                elif abc.upper() == 'AB':
                        if '0' in networks:
                            coalition.AB_C[0] = max(coalition.AB_C[0], networks['0']['best_valid_seen'])
                        if '1' in networks:
                            coalition.AB_C[1] = max(coalition.AB_C[1], networks['1']['best_valid_seen'])
                elif abc.upper() == 'AC':
                        if '0' in networks:
                            coalition.AC_B[0] = max(coalition.AC_B[0], networks['0']['best_valid_seen'])
                        if '2' in networks:
                            coalition.AC_B[2] = max(coalition.AC_B[2], networks['2']['best_valid_seen'])
                elif abc.upper() == 'BC':
                        if '1' in networks:
                            coalition.A_BC[1] = max(coalition.A_BC[1], networks['1']['best_valid_seen'])
                        if '2' in networks:
                            coalition.A_BC[2] = max(coalition.A_BC[2], networks['2']['best_valid_seen'])

                else:
                    continue
                # if C_size == 4000:
                #     print('after:', coalitions[key])
    return coalitions

def fix_solo_accuracies(coalition):
    SOLO_QUANTITY_A = .5004
    SOLO_QUANTITY_B = .6116
    SOLO_QUANTITY_C = .7091
    if coalition.partition == 'custom-quantity':
        coalition.A_BC[0] = SOLO_QUANTITY_A
        coalition.AC_B[1] = SOLO_QUANTITY_B
        coalition.AB_C[2] = SOLO_QUANTITY_C
        coalition.A_B_C_[0] = SOLO_QUANTITY_A
        coalition.A_B_C_[1] = SOLO_QUANTITY_B
    return coalition

def generate_coalition_table(i, coalition, filename, theta_max, is_uniform=True, is_squared=True, mean=1, sd=1):
    table_header = ['Coalition structure', "Client A's accuracy", "Client B's accuracy", "Client C's accuracy"]
    accuracies_as_table, reordered_profits, reordered_prices, profit_stability_dict, accuracy_stability_dict = createTableFromCoalition(coalition, theta_max, is_uniform=is_uniform, is_squared=is_squared, mean=mean, sd=sd)

    table_data = [
        (r"$\{A,B,C\}$", f"{coalition.ABC[0]*100:.2f}\\%", f"{coalition.ABC[1]*100:.2f}\\%", f"{coalition.ABC[2]*100:.2f}\\%"),
        (r"$\{A,B\}, \{C\}$", f"{coalition.AB_C[0]*100:.2f}\\%", f"{coalition.AB_C[1]*100:.2f}\\%", f"{coalition.AB_C[2]*100:.2f}\\%"),
        (r"$\{A,C\}, \{B\}$", f"{coalition.AC_B[0]*100:.2f}\\%", f"{coalition.AC_B[1]*100:.2f}\\%", f"{coalition.AC_B[2]*100:.2f}\\%"),
        (r"$\{B,C\}, \{A\}$", f"{coalition.A_BC[0]*100:.2f}\\%", f"{coalition.A_BC[1]*100:.2f}\\%", f"{coalition.A_BC[2]*100:.2f}\\%"),
        (r"$\{A\}, \{B\}, \{C\}$", f"{coalition.A_B_C_[0]*100:.2f}\\%", f"{coalition.A_B_C_[1]*100:.2f}\\%", f"{coalition.A_B_C_[2]*100:.2f}\\%")
    ]
    uniform_str = "True" if is_uniform else "False"
    table = (r"\subsection{Scenario " + str(i+1) + "}\n\n"
             r"\textbf{Simulation Setup}:" + "\n"
             r"C Dataset Size: {" + str(coalition.C_size) + "}\n"
             r"Theta Max: {" + str(theta_max) + "}\n"
             r"Is Uniform: {" + uniform_str + "}\n"
             r"Mean: {" + str(mean) + "}\n"
             r"SD: {" + str(sd) + "}\n\n"
             r"\textbf{Numerical results}: \n\n")
    if coalition.partition == 'noniid-labeldir':
        table += r"Dirichlet Beta Parameter: {" + str(coalition.beta) + "}\n"

    table += "\\begin{table}[h]\n\\centering\n\\caption{Training results.}\n\\label{training-results}\n\\begin{tabular}{|c|c|c|c|}\\hline\n"
    table += ' & '.join(table_header) + '\\\\ \\hline\n'
    for row in table_data:
        table += ' & '.join([str(cell) for cell in row]) + '\\\\ \\hline\n'
    table += '\\end{tabular}\n\\end{table}\n'

    print(reordered_prices)
    table_header = ['Coalition structure', "Client A's price", "Client B's price", "Client C's price"]
    table_data = [
        (r"$\{A,B,C\}$", f"{reordered_prices[0][0]:.2f}", f"{reordered_prices[0][1]:.2f}", f"{reordered_prices[0][2]:.2f}"),
        (r"$\{A,B\}, \{C\}$", f"{reordered_prices[1][0]:.2f}", f"{reordered_prices[1][1]:.2f}", f"{reordered_prices[1][2]:.2f}"),
        (r"$\{A,C\}, \{B\}$", f"{reordered_prices[2][0]:.2f}", f"{reordered_prices[2][1]:.2f}", f"{reordered_prices[2][2]:.2f}"),
        (r"$\{B,C\}, \{A\}$", f"{reordered_prices[3][0]:.2f}", f"{reordered_prices[3][1]:.2f}", f"{reordered_prices[3][2]:.2f}"),
        (r"$\{A\}, \{B\}, \{C\}$", f"{reordered_prices[4][0]:.2f}", f"{reordered_prices[4][1]:.2f}", f"{reordered_prices[4][2]:.2f}")
    ]
    table += '\n\n'
    table += "\\begin{table}[h]\n\\centering\n\\caption{Price results.}\n\\label{price-results}\n\\begin{tabular}{|c|c|c|c|}\\hline\n"
    table += ' & '.join(table_header) + '\\\\ \\hline\n'
    for row in table_data:
        table += ' & '.join([str(cell) for cell in row]) + '\\\\ \\hline\n'
    table += '\\end{tabular}\n\\end{table}\n'

    table_header = ['Coalition structure', "Client A's profit", "Client B's price", "Client C's price"]
    table_data = [
        (r"$\{A,B,C\}$", f"{reordered_profits[0][0]:.2f}", f"{reordered_prices[0][1]:.2f}", f"{reordered_prices[0][2]:.2f}"),
        (r"$\{A,B\}, \{C\}$", f"{reordered_profits[1][0]:.2f}", f"{reordered_prices[1][1]:.2f}", f"{reordered_prices[1][2]:.2f}"),
        (r"$\{A,C\}, \{B\}$", f"{reordered_profits[2][0]:.2f}", f"{reordered_prices[2][1]:.2f}", f"{reordered_prices[2][2]:.2f}"),
        (r"$\{B,C\}, \{A\}$", f"{reordered_profits[3][0]:.2f}", f"{reordered_prices[3][1]:.2f}", f"{reordered_prices[3][2]:.2f}"),
        (r"$\{A\}, \{B\}, \{C\}$", f"{reordered_profits[4][0]:.2f}", f"{reordered_prices[4][1]:.2f}", f"{reordered_prices[4][2]:.2f}")
    ]
    table += '\n\n'
    table += "\\begin{table}[h]\n\\centering\n\\caption{Profit results.}\n\\label{profit-results}\n\\begin{tabular}{|c|c|c|c|}\\hline\n"
    table += ' & '.join(table_header) + '\\\\ \\hline\n'
    for row in table_data:
        table += ' & '.join([str(cell) for cell in row]) + '\\\\ \\hline\n'
    table += '\\end{tabular}\n\\end{table}\n'

    table += r"\textbf{Core stable coalition structures}:" + "\n"
    table += r"\begin{itemize}" + "\n"

    # Competitive coalition structures
    coalitions = []
    for key, value in profit_stability_dict.items():
        if 'True' in value:
            if key == 'A_BC':
                coalitions.append(r'$\{B,C\}, \{A\}$')
            elif key == 'AB_C':
                coalitions.append(r'$\{A,B\}, \{C\}$')
            elif key == 'AC_B':
                coalitions.append(r'$\{A,C\}, \{B\}$')
            elif key == 'A_B_C_':
                coalitions.append(r'$\{A\}, \{B\}, \{C\}$')
            else:  # key is ABC
                coalitions.append(r'$\{A,B,C\}$')
    coalitions_str = ', '.join(coalitions)
    result_str = r'\item Competitive: ' + coalitions_str
    table += result_str + "\n"

    # Non-competitive coalition structures
    coalitions = []
    for key, value in accuracy_stability_dict.items():
        if 'True' in value:
            if key == 'A_BC':
                coalitions.append(r'$\{B,C\}, \{A\}$')
            elif key == 'AB_C':
                coalitions.append(r'$\{A,B\}, \{C\}$')
            elif key == 'AC_B':
                coalitions.append(r'$\{A,C\}, \{B\}$')
            elif key == 'A_B_C_':
                coalitions.append(r'$\{A\}, \{B\}, \{C\}$')
            else:  # key is ABC
                coalitions.append(r'$\{A,B,C\}$')
    coalitions_str = ', '.join(coalitions)
    result_str = r'\item Non-competitive: ' + coalitions_str
    table += result_str + "\n"

    table += r"\end{itemize}" + "\n"

    with open(filename, 'w') as f:
        f.write(table)

if __name__ == '__main__':
    args = get_args()
    print('--- Parsing Logs ---')
    coalitions = parse_logs()

    coalition_str_dict = {k: str(v) for k, v in coalitions.items()}
    print('After parsing logs:')
    for k, v in coalition_str_dict.items():
        print(k, v)

    print('--- Parsing FT Logs ---')
    coalitions = parse_ft_logs(coalitions)
    for coalition in coalitions:
        coalitions[coalition] = fix_solo_accuracies(coalitions[coalition])

    coalition_str_dict = {k: str(v) for k, v in coalitions.items()}
    print('After parsing FT logs:')
    for k, v in coalition_str_dict.items():
        print(k, v)

    for i, coalition in enumerate(coalitions):
        generate_coalition_table(
            i,
            coalitions[coalition],
            f'{coalition[0]}_Coalition.txt',
            theta_max = 10000,
            is_uniform=True,
            mean=1,
            sd=1,)