import numpy as np
import json
import argparse
import logging
import os
import re
from math import *
import dill as pickle
from itertools import product
import sys
from importlib import reload
import bestresponse
reload(bestresponse)
from bestresponse import createTableFromCoalition, Coalition, calculate_equilibrium_price, calculate_equilibrium_profits

LOGS_PATH = './logs/'
LOGS_FT_PATH = './logs_ft/'
PARAMETER_PATTERN = r"(?=.*C_size=(\d+))(?=.*abc='([^']+)')(?=.*beta=([\d.]+))(?=.*\spartition='([^']+)')"

BETA_MAP = {
    "0001": "0.001",
    "001": "0.01",
    "01": "0.1",
    "10": "1.0",
    "100": "10.0",
    "1000": "100.0",
}
debug = True

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
                if debug:
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
                    partition = match.group(4)
                # extract the score value using regular expressions
                match = re.search(r"New best score: ([\d.]+)", last_line)
                score = None
                if match:
                    score = float(match.group(1))
                    # print(f"C_size={C_size}, abc={abc}, score={score}")

                if C_size is None or score is None:
                    raise ValueError("Improper format of log file")
                key = (partition, C_size, beta)
                value = [score] * 3
                # print(key, value)
                if key not in coalitions:
                    # create a new coalition object with the updated property
                    if abc.upper() == "ABC":
                        coalition = Coalition(C_size, value, [], [], [], [0]*3, beta, partition=partition)
                    elif abc.upper() == 'AB':
                        coalition = Coalition(C_size, [], value, [], [], [0]*3, beta, partition=partition)
                    elif abc.upper() == 'AC':
                        coalition = Coalition(C_size, [], [], value, [], [0]*3, beta, partition=partition)
                    elif abc.upper() == 'BC':
                        coalition = Coalition(C_size, [], [], [], value, [0]*3, beta, partition=partition)
                    elif abc.upper() == 'A':
                        # In these single client cases, set values that aren't read
                        # to be 0, so they are overriden later with max() function
                        value[1] = 0
                        value[2] = 0
                        coalition = Coalition(C_size, [], [], [], [], value, beta, partition=partition)
                    elif abc.upper() == 'B':
                        value[0] = 0
                        value[2] = 0
                        coalition = Coalition(C_size, [], [], [], [], value, beta, partition=partition)
                    elif abc.upper() == 'C':
                        value[0] = 0
                        value[1] = 0
                        coalition = Coalition(C_size, [], [], [], [], value, beta, partition=partition)
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
                if debug:
                    print(f'Reading: {fname}')
                parts = fname.split('-')
                abc = parts[0].upper()
                abc = abc.split('/')[-1]
                partition = str(parts[1] + '-' + parts[2])
                C_size = int(parts[3])
                beta = parts[4]
                beta = BETA_MAP[beta]
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
                key = (partition, C_size, beta)
                values = [network['best_valid_seen'] for network in networks.values()]
                # if C_size == 10000:
                    # print('Networks:', networks)
                    # print('values:', values)
                #     print('before:', coalitions[key])
                coalition = coalitions[key]
                if debug:
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
    if coalition.C_size == 8000:
        SOLO_QUANTITY_C = .7091
    else:
        SOLO_QUANTITY_C = coalition.A_B_C_[2]
    if coalition.partition == 'custom-quantity':
        coalition.A_BC[0] = SOLO_QUANTITY_A
        coalition.AC_B[1] = SOLO_QUANTITY_B
        coalition.AB_C[2] = SOLO_QUANTITY_C
        coalition.A_B_C_[0] = SOLO_QUANTITY_A
        coalition.A_B_C_[1] = SOLO_QUANTITY_B
    elif coalition.partition == 'noniid-labeldir':
        SOLO_QUANTITY_A = coalition.A_B_C_[0]
        SOLO_QUANTITY_B = coalition.A_B_C_[1]
        SOLO_QUANTITY_C = coalition.A_B_C_[2]
        coalition.A_BC[0] = SOLO_QUANTITY_A
        coalition.AC_B[1] = SOLO_QUANTITY_B
        coalition.AB_C[2] = SOLO_QUANTITY_C

    return coalition

def generate_table_text(base_accuracies_array, category="Competitive"):
    coalitions = []
    for key, value in base_accuracies_array.items():
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
    result_str = r'\item ' + category + ': ' + coalitions_str
    return result_str + "\n"

def generate_coalition_table(i, coalition, theta_max, filename=None, is_uniform=True, is_squared=True, mean=1, sd=1):
    table_header = ['Coalition structure', "Client A's accuracy", "Client B's accuracy", "Client C's accuracy"]
    uniform_str = "True" if is_uniform else "False"
    squared_str = "True" if is_squared else "False"
    if filename is None:
        beta_string = str(coalition.beta).replace('.', '')
        filename = f'{coalition.partition}_beta-{beta_string}_csize-{coalition.C_size}_thetamax-{theta_max}_uniform-{uniform_str}_mean-{mean}_sd-{sd}_squared-{squared_str}.txt'

    accuracies_as_table, degredaded_accuracies_as_table, reordered_profits, reordered_prices, degredaded_reordered_profits, degredaded_reordered_prices, base_accuracies_array, degraded_accuracies_array = createTableFromCoalition(coalition, theta_max, is_uniform=is_uniform, is_squared=is_squared, mean=mean, sd=sd)
    partition_str = "Non-IID Label Dirichlet" if coalition.partition == "noniid-labeldir" else "IID"
    table = (r"\subsection{Scenario " + str(i+1) + "}\n\n"
             r"\textbf{Simulation Setup}:" + "\n"
             r"Data Scenario: " + partition_str + "\n"
             r"C Dataset Size: {" + str(coalition.C_size) + "}\n"
             r"Theta Max: {" + str(theta_max) + "}\n"
             r"Is Squared: {" + squared_str + "}\n"
             r"Is Uniform: {" + uniform_str + "}\n")

    if coalition.partition == 'noniid-labeldir':
        table += r"Dirichlet Beta Parameter: {" + str(coalition.beta) + "}\n"
    if not is_uniform:
        table += r"Mean: {" + str(mean) + "}\n"
        table += r"SD: {" + str(sd) + "}\n\n"
    else:
        table += "\n"
    table += "\\subsection{Scenario 1: Without Degredation}\n"
    table += r"\textbf{Numerical results}:" + "\n\n"

    table += "\\begin{table}[h]\n\\centering\n\\caption{Training results.}\n\\label{training-results}\n\\begin{tabular}{|c|c|c|c|}\\hline\n"
    table += ' & '.join(table_header) + '\\\\ \\hline\n'
    table_data = [
        (r"$\{A,B,C\}$", f"{coalition.ABC[0]*100:.2f}\\%", f"{coalition.ABC[1]*100:.2f}\\%", f"{coalition.ABC[2]*100:.2f}\\%"),
        (r"$\{A,B\}, \{C\}$", f"{coalition.AB_C[0]*100:.2f}\\%", f"{coalition.AB_C[1]*100:.2f}\\%", f"{coalition.AB_C[2]*100:.2f}\\%"),
        (r"$\{A,C\}, \{B\}$", f"{coalition.AC_B[0]*100:.2f}\\%", f"{coalition.AC_B[1]*100:.2f}\\%", f"{coalition.AC_B[2]*100:.2f}\\%"),
        (r"$\{B,C\}, \{A\}$", f"{coalition.A_BC[0]*100:.2f}\\%", f"{coalition.A_BC[1]*100:.2f}\\%", f"{coalition.A_BC[2]*100:.2f}\\%"),
        (r"$\{A\}, \{B\}, \{C\}$", f"{coalition.A_B_C_[0]*100:.2f}\\%", f"{coalition.A_B_C_[1]*100:.2f}\\%", f"{coalition.A_B_C_[2]*100:.2f}\\%")
    ]
    for row in table_data:
        table += ' & '.join([str(cell) for cell in row]) + '\\\\ \\hline\n'
    table += '\\end{tabular}\n\\end{table}\n'

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

    table_header = ['Coalition structure', "Client A's profit", "Client B's profit", "Client C's profit"]
    table_data = [
        (r"$\{A,B,C\}$", f"{reordered_profits[0][0]:.2f}", f"{reordered_profits[0][1]:.2f}", f"{reordered_profits[0][2]:.2f}"),
        (r"$\{A,B\}, \{C\}$", f"{reordered_profits[1][0]:.2f}", f"{reordered_profits[1][1]:.2f}", f"{reordered_profits[1][2]:.2f}"),
        (r"$\{A,C\}, \{B\}$", f"{reordered_profits[2][0]:.2f}", f"{reordered_profits[2][1]:.2f}", f"{reordered_profits[2][2]:.2f}"),
        (r"$\{B,C\}, \{A\}$", f"{reordered_profits[3][0]:.2f}", f"{reordered_profits[3][1]:.2f}", f"{reordered_profits[3][2]:.2f}"),
        (r"$\{A\}, \{B\}, \{C\}$", f"{reordered_profits[4][0]:.2f}", f"{reordered_profits[4][1]:.2f}", f"{reordered_profits[4][2]:.2f}")
    ]
    table += '\n\n'
    table += "\\begin{table}[h]\n\\centering\n\\caption{Profit results.}\n\\label{profit-results}\n\\begin{tabular}{|c|c|c|c|}\\hline\n"
    table += ' & '.join(table_header) + '\\\\ \\hline\n'
    for row in table_data:
        table += ' & '.join([str(cell) for cell in row]) + '\\\\ \\hline\n'
    table += '\\end{tabular}\n\\end{table}\n'

    table += r"\textbf{Core stable coalition structures}:" + "\n"
    table += r"\begin{itemize}" + "\n"

    table += generate_table_text(base_accuracies_array[0], 'Competitive')
    table += generate_table_text(base_accuracies_array[1], 'Non-competitive')

    table += r"\end{itemize}" + "\n"

    table += r"\textbf{Individual stable coalition structures}:" + "\n"
    table += r"\begin{itemize}" + "\n"

    table += generate_table_text(base_accuracies_array[2], 'Competitive')
    table += generate_table_text(base_accuracies_array[3], 'Non-competitive')

    table += r"\end{itemize}" + "\n"

    table += "\\subsection{Scenario 2: With Degredation}\n"

    table_header = ['Coalition structure', "Client A's accuracy", "Client B's accuracy", "Client C's accuracy"]
    table_data = [
        (r"$\{A,B,C\}$", f"{degredaded_accuracies_as_table[0][0]*100:.2f}\\%", f"{degredaded_accuracies_as_table[0][1]*100:.2f}\\%", f"{degredaded_accuracies_as_table[0][2]*100:.2f}\\%"),
        (r"$\{A,B\}, \{C\}$", f"{degredaded_accuracies_as_table[1][0]*100:.2f}\\%", f"{degredaded_accuracies_as_table[1][1]*100:.2f}\\%", f"{degredaded_accuracies_as_table[1][2]*100:.2f}\\%"),
        (r"$\{A,C\}, \{B\}$", f"{degredaded_accuracies_as_table[2][0]*100:.2f}\\%", f"{degredaded_accuracies_as_table[2][1]*100:.2f}\\%", f"{degredaded_accuracies_as_table[2][2]*100:.2f}\\%"),
        (r"$\{B,C\}, \{A\}$", f"{degredaded_accuracies_as_table[3][0]*100:.2f}\\%", f"{degredaded_accuracies_as_table[3][1]*100:.2f}\\%", f"{degredaded_accuracies_as_table[3][2]*100:.2f}\\%"),
        (r"$\{A\}, \{B\}, \{C\}$", f"{degredaded_accuracies_as_table[4][0]*100:.2f}\\%", f"{degredaded_accuracies_as_table[4][1]*100:.2f}\\%", f"{degredaded_accuracies_as_table[4][2]*100:.2f}\\%")
    ]
    table += '\n\n'
    table += "\\begin{table}[h]\n\\centering\n\\caption{Accuracy results for the 'degredated' scenario.}\n\\label{accuracy-results-degredated}\n\\begin{tabular}{|c|c|c|c|}\\hline\n"
    table += ' & '.join(table_header) + '\\\\ \\hline\n'
    for row in table_data:
        table += ' & '.join([str(cell) for cell in row]) + '\\\\ \\hline\n'
    table += '\\end{tabular}\n\\end{table}\n'

    table_header = ['Coalition structure', "Client A's price", "Client B's price", "Client C's price"]
    table_data = [
        (r"$\{A,B,C\}$", f"{degredaded_reordered_prices[0][0]:.2f}", f"{degredaded_reordered_prices[0][1]:.2f}", f"{degredaded_reordered_prices[0][2]:.2f}"),
        (r"$\{A,B\}, \{C\}$", f"{degredaded_reordered_prices[1][0]:.2f}", f"{degredaded_reordered_prices[1][1]:.2f}", f"{degredaded_reordered_prices[1][2]:.2f}"),
        (r"$\{A,C\}, \{B\}$", f"{degredaded_reordered_prices[2][0]:.2f}", f"{degredaded_reordered_prices[2][1]:.2f}", f"{degredaded_reordered_prices[2][2]:.2f}"),
        (r"$\{B,C\}, \{A\}$", f"{degredaded_reordered_prices[3][0]:.2f}", f"{degredaded_reordered_prices[3][1]:.2f}", f"{degredaded_reordered_prices[3][2]:.2f}"),
        (r"$\{A\}, \{B\}, \{C\}$", f"{degredaded_reordered_prices[4][0]:.2f}", f"{degredaded_reordered_prices[4][1]:.2f}", f"{degredaded_reordered_prices[4][2]:.2f}")
    ]
    table += '\n\n'
    table += "\\begin{table}[h]\n\\centering\n\\caption{Price results.}\n\\label{price-results}\n\\begin{tabular}{|c|c|c|c|}\\hline\n"
    table += ' & '.join(table_header) + '\\\\ \\hline\n'
    for row in table_data:
        table += ' & '.join([str(cell) for cell in row]) + '\\\\ \\hline\n'
    table += '\\end{tabular}\n\\end{table}\n'

    table_header = ['Coalition structure', "Client A's profit", "Client B's profit", "Client C's profit"]
    table_data = [
        (r"$\{A,B,C\}$", f"{degredaded_reordered_profits[0][0]:.2f}", f"{degredaded_reordered_profits[0][1]:.2f}", f"{degredaded_reordered_profits[0][2]:.2f}"),
        (r"$\{A,B\}, \{C\}$", f"{degredaded_reordered_profits[1][0]:.2f}", f"{degredaded_reordered_profits[1][1]:.2f}", f"{degredaded_reordered_profits[1][2]:.2f}"),
        (r"$\{A,C\}, \{B\}$", f"{degredaded_reordered_profits[2][0]:.2f}", f"{degredaded_reordered_profits[2][1]:.2f}", f"{degredaded_reordered_profits[2][2]:.2f}"),
        (r"$\{B,C\}, \{A\}$", f"{degredaded_reordered_profits[3][0]:.2f}", f"{degredaded_reordered_profits[3][1]:.2f}", f"{degredaded_reordered_profits[3][2]:.2f}"),
        (r"$\{A\}, \{B\}, \{C\}$", f"{degredaded_reordered_profits[4][0]:.2f}", f"{degredaded_reordered_profits[4][1]:.2f}", f"{degredaded_reordered_profits[4][2]:.2f}")
    ]
    table += '\n\n'
    table += "\\begin{table}[h]\n\\centering\n\\caption{Profit results.}\n\\label{profit-results}\n\\begin{tabular}{|c|c|c|c|}\\hline\n"
    table += ' & '.join(table_header) + '\\\\ \\hline\n'
    for row in table_data:
        table += ' & '.join([str(cell) for cell in row]) + '\\\\ \\hline\n'
    table += '\\end{tabular}\n\\end{table}\n'

    table += r"\textbf{Core stable coalition structures}:" + "\n"
    table += r"\begin{itemize}" + "\n"
    table += generate_table_text(degraded_accuracies_array[0], 'Competitive')
    table += generate_table_text(degraded_accuracies_array[1], 'Non-competitive')
    table += r"\end{itemize}" + "\n"

    table += r"\textbf{Individual stable coalition structures}:" + "\n"
    table += r"\begin{itemize}" + "\n"
    table += generate_table_text(degraded_accuracies_array[2], 'Competitive')
    table += generate_table_text(degraded_accuracies_array[3], 'Non-competitive')

    table += r"\end{itemize}" + "\n"

    with open(f'./latex/{filename}', 'w') as f:
        f.write(table)

if __name__ == '__main__':
    args = get_args()
    print('--- Parsing Logs ---')
    coalitions = parse_logs()

    coalition_str_dict = {k: str(v) for k, v in coalitions.items()}
    if debug:
        print('After parsing logs:')
        for k, v in coalition_str_dict.items():
            print(k, v)

    print('--- Parsing FT Logs ---')
    coalitions = parse_ft_logs(coalitions)
    for coalition in coalitions:
        coalitions[coalition] = fix_solo_accuracies(coalitions[coalition])

    coalition_str_dict = {k: str(v) for k, v in coalitions.items()}
    if debug:
        print('After parsing FT logs:')
        for k, v in coalition_str_dict.items():
            print(k, v)
    THETA_MAX = 10000
    IS_UNIFORM = False
    IS_SQUARED = True
    MEAN = 5000
    SD = 2000
    for i, coalition in enumerate(coalitions):
        generate_coalition_table(
            i,
            coalitions[coalition],
            theta_max = THETA_MAX,
            is_uniform=IS_UNIFORM,
            is_squared=IS_SQUARED,
            mean=MEAN,
            sd=SD,)

    SOLO_QUANTITY_A = .5004
    SOLO_QUANTITY_B = .6116
    SOLO_QUANTITY_C = .7091
    ABC_Quantity = [.8803, .8821, .8817]
    AB_C_Quantity = [.7976, .8007, SOLO_QUANTITY_C]
    AC_B_Quantity = [0.8550, SOLO_QUANTITY_B, .8608]
    A_BC_Quantity = [SOLO_QUANTITY_A, .8732, .8762]
    A_B_C_Quantity = [SOLO_QUANTITY_A, SOLO_QUANTITY_B, SOLO_QUANTITY_C]
    print('price:', calculate_equilibrium_price(SOLO_QUANTITY_C, SOLO_QUANTITY_B, SOLO_QUANTITY_A, 10000))
    print('profits:', calculate_equilibrium_profits(SOLO_QUANTITY_C, SOLO_QUANTITY_B, SOLO_QUANTITY_A, 10000))
    quantity_coalition = Coalition(8000, ABC_Quantity, AB_C_Quantity, AC_B_Quantity, A_BC_Quantity, A_B_C_Quantity, 0.1)
    generate_coalition_table(10, 
                            quantity_coalition,
                            theta_max = THETA_MAX,
                            is_uniform=IS_UNIFORM,
                            is_squared=IS_SQUARED,
                            mean=MEAN,
                            sd=SD,)
    sys.exit()
    # Non-iid
    SOLO_DIRICHLET_A = .6085
    SOLO_DIRICHLET_B = .6394
    SOLO_DIRICHLET_C = .5932
    ABC_Dirichlet = [.8681, .8668, .8655]
    AB_C_Dirichlet = [.8440, .8453, SOLO_DIRICHLET_C]
    AC_B_Dirichlet = [0.8301, SOLO_DIRICHLET_B, .8359]
    A_BC_Dirichlet = [SOLO_DIRICHLET_A, .8347, .8320]
    A_B_C_Dirichlet = [SOLO_DIRICHLET_A, SOLO_DIRICHLET_B, SOLO_DIRICHLET_C]

    dirichlet_coalition = Coalition(2480, ABC_Dirichlet, AB_C_Dirichlet, AC_B_Dirichlet, A_BC_Dirichlet, A_B_C_Dirichlet, 0.1, partition='noniid-labeldir')

    generate_coalition_table(10, 
                            dirichlet_coalition, 
                            theta_max = THETA_MAX,
                            is_uniform=IS_UNIFORM,
                            is_squared=IS_SQUARED,
                            mean=MEAN,
                            sd=SD,)
