import argparse
import os
import warnings

# import jax.numpy as jnp
# from jax import grad
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import truncnorm

from scipy.optimize import basinhopping
import dill as pickle

class Coalition:
    def __init__(self, C_size, ABC, AB_C, AC_B, A_BC, A_B_C_, beta, partition='custom-quantity'):
        self.C_size = C_size
        self.ABC = ABC
        self.AB_C = AB_C
        self.AC_B = AC_B
        self.A_BC = A_BC
        self.A_B_C_ = A_B_C_
        self.beta = beta
        self.partition = partition

    def __str__(self):
        return f"Coalition(C_size={self.C_size}, ABC={self.ABC}, AB_C={self.AB_C}, AC_B={self.AC_B}, A_BC={self.A_BC}, A_B_C_={self.A_B_C_}, beta={self.beta})"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calculate', default=False, required=False, action='store_true', help='whether to calculate price values instead of loading from pickle')
    args = parser.parse_args()
    return args

A = [0.9, 0.8, 0.7]
# Custom Quantity

SOLO_QUANTITY_A = (0, .5004)
SOLO_QUANTITY_B = (1, .6116)
SOLO_QUANTITY_C = (2, .7091)
ABC_Quantity = [(0, .8803), (1, .8821), (2, .8817)]
AB_C_Quantity = [(0, .7976), (1, .8007), SOLO_QUANTITY_C]
AC_B_Quantity = [(0, 0.8550), SOLO_QUANTITY_B, (2, .8608)]
A_BC_Quantity = [SOLO_QUANTITY_A, (1, .8732), (2, .8762)]
A_B_C_Quantity = [SOLO_QUANTITY_A, SOLO_QUANTITY_B, SOLO_QUANTITY_C]

quantity_coalition = Coalition(8000, ABC_Quantity, AB_C_Quantity, AC_B_Quantity, A_BC_Quantity, A_B_C_Quantity, 0.1)

# Non-iid
SOLO_DIRICHLET_A = (0, .6085)
SOLO_DIRICHLET_B = (1, .6394)
SOLO_DIRICHLET_C = (2, .5932)
ABC_Dirichlet = [(0, .8681), (1, .8668), (2, .8655)]
AB_C_Dirichlet = [(0, .8440), (1, .8453), SOLO_DIRICHLET_C]
AC_B_Dirichlet = [(0, 0.8301), SOLO_DIRICHLET_B, (2, .8359)]
A_BC_Dirichlet = [SOLO_DIRICHLET_A, (1, .8347), (2, .8320)]
A_B_C_Dirichlet = [SOLO_DIRICHLET_A, SOLO_DIRICHLET_B, SOLO_DIRICHLET_C]

dirichlet_coalition = Coalition(2480, ABC_Dirichlet, AB_C_Dirichlet, AC_B_Dirichlet, A_BC_Dirichlet, A_B_C_Dirichlet, 0.1, partition='noniid-labeldir')

C_pri = [0.5, 0.3, 0.1]

def sigma(m, n, p, A, is_squared):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if is_squared:
            return (p[m] - p[n]) / (A[m]**2 - A[n]**2)
        return (p[m] - p[n]) / (A[m] - A[n])
    

def N(theta, mean, sd, theta_max, is_uniform=False):
    if theta < 0:
        return 0
    if theta >= theta_max:
        return 1
    if is_uniform:
        return theta / theta_max
    else:
        a, b = (0 - mean) / sd, (theta_max - mean) / sd
        return truncnorm.cdf(theta, a, b, loc=mean, scale=sd)# / (truncnorm.cdf(theta_max, a, b, loc=mean, scale=sd) - truncnorm.cdf(0, a, b, loc=mean, scale=sd))
# # create an array of theta values to plot
# theta_max = 10000
# mean = 5000
# sd = 500
# theta_vals = np.linspace(-10, theta_max+10, 1000)

# # calculate the corresponding PDF values
# pdf_vals = [N(theta, mean, sd, theta_max, is_uniform=True) for theta in theta_vals]

# # create the plot
# fig, ax = plt.subplots()
# ax.plot(theta_vals, pdf_vals, label='Truncated normal distribution')
# ax.set_xlabel('Theta')
# ax.set_ylabel('PDF')
# ax.legend()
# plt.show()
def W0(p, A, mean, sd, theta_max, is_uniform, is_squared):
    try:
        sigma_0_1 = sigma(0, 1, p, A, is_squared)
        sigma_0_2 = sigma(0, 2, p, A, is_squared)
        return p[0] * (1 - N(max(sigma_0_1, sigma_0_2), mean, sd, theta_max, is_uniform))

    except ZeroDivisionError:
        return 0

def W1(p, A, mean, sd, theta_max, is_uniform, is_squared):
    try:
        sigma_0_1 = sigma(0, 1, p, A, is_squared)
        sigma_1_2 = sigma(1, 2, p, A, is_squared)
        return p[1] * N(sigma_0_1 - sigma_1_2, mean, sd, theta_max, is_uniform)

    except ZeroDivisionError:
        return 0

def W2(p, A, mean, sd, theta_max, is_uniform, is_squared):
    try:
        sigma_1_2 = sigma(1, 2, p, A, is_squared)
        sigma_0_2 = sigma(0, 2, p, A, is_squared)
        return p[2] * N(min(sigma_1_2, sigma_0_2), mean, sd, theta_max, is_uniform)

    except ZeroDivisionError:
        return 0

def W0Obj(p, A, mean, sd, theta_max, is_uniform, is_squared):
    return -W0(p, A, mean, sd, theta_max, is_uniform, is_squared)

def W1Obj(p, A, mean, sd, theta_max, is_uniform, is_squared):
    return -W1(p, A, mean, sd, theta_max, is_uniform, is_squared)

def W2Obj(p, A, mean, sd, theta_max, is_uniform, is_squared):
    return -W2(p, A, mean, sd, theta_max, is_uniform, is_squared)

def update_price(i, p, A, mean, sd, theta_max, is_uniform, is_squared):
    if i == 0:
        res = basinhopping(lambda x: W0Obj([x, p[1], p[2]], A, mean, sd, theta_max, is_uniform, is_squared), x0=p[0], minimizer_kwargs={'method': 'BFGS'})
        return [res.x[0], p[1], p[2]]
    elif i == 1:
        res = basinhopping(lambda x: W1Obj([p[0], x, p[2]], A, mean, sd, theta_max, is_uniform, is_squared), x0=p[1], minimizer_kwargs={'method': 'BFGS'})
        return [p[0], res.x[0], p[2]]
    elif i == 2:
        res = basinhopping(lambda x: W2Obj([p[0], p[1], x], A, mean, sd, theta_max, is_uniform, is_squared), x0=p[2], minimizer_kwargs={'method': 'BFGS'})
        return [p[0], p[1], res.x[0]]

text_name = ['ABC', 'AB_C', 'AC_B', 'A_BC', 'A_B_C_']
quantity_arrays = [ABC_Quantity, AB_C_Quantity, AC_B_Quantity, A_BC_Quantity, A_B_C_Quantity]
dirichlet_arrays = [ABC_Dirichlet, AB_C_Dirichlet, AC_B_Dirichlet, A_BC_Dirichlet, A_B_C_Dirichlet]

def get_profit(i, price, scores, mean, sd, theta_max, is_uniform, is_squared):
    if i == 0:
        return W0(price, scores, mean, sd, theta_max, is_uniform, is_squared)
    elif i == 1:
        return W1(price, scores, mean, sd, theta_max, is_uniform, is_squared)
    elif i == 2:
        return W2(price, scores, mean, sd, theta_max, is_uniform, is_squared)

def optimize(j, partition, mean, sd, theta_max, is_uniform, is_squared):
    print(partition)
    if isinstance(partition[0], tuple):
        # partition is already a list of tuples
        pass
    else:
        # partition is a list of single values, convert to a list of tuples
        partition = [(i, x) for i, x in enumerate(partition)]

    def set_duplicate_values_to_zero(arr):
        if arr[0][1] == arr[1][1] == arr[2][1]:
            arr[0] = (arr[0][0], 0)
            arr[1] = (arr[1][0], 0)
            arr[2] = (arr[2][0], 0)
        else:
            if arr[0][1] == arr[1][1]:
                arr[0] = (arr[0][0], 0)
                arr[1] = (arr[1][0], 0)
            if arr[1][1] == arr[2][1]:
                arr[1] = (arr[1][0], 0)
                arr[2] = (arr[2][0], 0)
            if arr[0][1] == arr[2][1]:
                arr[0] = (arr[0][0], 0)
                arr[2] = (arr[2][0], 0)
        return arr
    
    partition = set_duplicate_values_to_zero(partition)
    partition = sorted(partition, key=lambda x: x[1], reverse=True)
    p_init = [0]*3
    for i, (_, v) in enumerate(partition):
        p_init[i] = 5 if v != 0 else 0
    p_new = p_init.copy()
    ordering = []
    scores = []
    for ord, score in partition:
        ordering += [ord]
        scores += [score]
    maxsteps=100
    steps=0
    while True:
        steps+=1
        for i in range(3):
            p_new = update_price(float(i), p_new, scores, mean, sd, theta_max, is_uniform, is_squared)
        if np.allclose(np.array(p_init), np.array(p_new), rtol=1e-8) or steps>maxsteps:
            print(f'steps: {steps}')
            break

        p_init = p_new.copy()

    print(f'Optimal prices: {p_new} with order {ordering} for partition {text_name[j]}')
    profits = []
    for i in range(3):
        profits.append(get_profit(i, p_new, scores, mean, sd, theta_max, is_uniform, is_squared))
    profits = np.nan_to_num(np.array(profits), nan=0).tolist()
    print(f'Profits: {profits}')
    return p_new, profits, ordering, j

def get_final_table(custom_array):
    profit_table = []
    prices_table = []
    for i, partition in enumerate(text_name):
        prices, profits, ordering, _ = custom_array[i]
        profits = np.array(profits)
        # print(f'profits before: {profits} ordering: {ordering}')
        profits_by_ordering = [0] * 3
        for i, order in enumerate(ordering):
            profits_by_ordering[order] = profits[i]
        # print(f'profits after: {profits_by_ordering}')
        # print(f'prices before: {prices} ordering: {ordering}')
        prices_by_ordering = [0] * 3
        for i, order in enumerate(ordering):
            prices_by_ordering[order] = prices[i]
        # print(f'prices after: {prices_by_ordering}')
        profit_table.append(profits_by_ordering)
        prices_table.append(prices_by_ordering)
    return np.array(profit_table), np.array(prices_table)

tdict = {text_name[i]: i for i in range(len(text_name))}
cdict = {'A': 0, 'B': 1, 'C': 2}
def test_ABC_stability(table):
    A_current = table[tdict['ABC']][cdict['A']]
    B_current = table[tdict['ABC']][cdict['B']]
    C_current = table[tdict['ABC']][cdict['C']]
    if table[tdict['AB_C']][cdict['A']] > A_current and table[tdict['AB_C']][cdict['B']] > B_current:
        return (False, 'Not stable due to AB_C')
    
    # Check coalition {A,B}{C}
    if table[tdict['A_BC']][cdict['B']] > B_current and table[tdict['A_BC']][cdict['C']] > C_current:
        return (False, 'Not stable due to AB_C')
    
    # Check coalition {A,C}{B}
    if table[tdict['AC_B']][cdict['A']] > A_current and table[tdict['AC_B']][cdict['C']] > C_current:
        return (False, 'Not stable due to AB_C')
    
    # Check coalition {A}
    if table[tdict['A_B_C_']][cdict['A']] > A_current:
        return (False, 'Not stable due to A in A_B_C_')
    
    # Check coalition {B}
    if table[tdict['A_B_C_']][cdict['B']] > B_current:
        return (False, 'Not stable due to B in A_B_C_')
    
    # Check coalition {C}
    if table[tdict['A_B_C_']][cdict['C']] > C_current:
        return (False, 'Not stable due to C in A_B_C_')
    
    return (True, 'Core stable')

def test_AB_C_stability(table):
    A_current = table[tdict['AB_C']][cdict['A']]
    B_current = table[tdict['AB_C']][cdict['B']]
    C_current = table[tdict['AB_C']][cdict['C']]
    
    # Check coalition {A,B,C}
    if table[tdict['ABC']][cdict['A']] > A_current and table[tdict['ABC']][cdict['B']] > B_current and table[tdict['ABC']][cdict['C']] > C_current:
        return (False, 'Not stable due to ABC')
    
    # Check coalition {A,C}{B}
    if table[tdict['AC_B']][cdict['A']] > A_current and table[tdict['AC_B']][cdict['C']] > C_current:
        return (False, 'Not stable due to AC_B')
    
    # Check coalition {A}{B,C}
    if table[tdict['A_BC']][cdict['B']] > B_current and table[tdict['A_BC']][cdict['C']] > C_current:
        return (False, 'Not stable due to B in A_BC')
    
    # Check coalition {A}
    if table[tdict['A_B_C_']][cdict['A']] > A_current:
        return (False, 'Not stable due to A in A_B_C_')
    
    # Check coalition {B}
    if table[tdict['A_B_C_']][cdict['B']] > B_current:
        return (False, 'Not stable due to B in A_B_C_')
    
    return (True, 'Core stable')

def test_AC_B_stability(table):
    A_current = table[tdict['AC_B']][cdict['A']]
    B_current = table[tdict['AC_B']][cdict['B']]
    C_current = table[tdict['AC_B']][cdict['C']]
    
    # Check coalition {A,B,C}
    if table[tdict['ABC']][cdict['A']] > A_current and table[tdict['ABC']][cdict['B']] > B_current and table[tdict['ABC']][cdict['C']] > C_current:
        return (False, 'Not stable due to ABC')

    # Check coalition {A,B}
    if table[tdict['AB_C']][cdict['A']] > A_current and table[tdict['AB_C']][cdict['B']] > B_current:
        return (False, 'Not stable due to AB_C')
    
    # Check coalition {A,C}
    if table[tdict['A_BC']][cdict['B']] > B_current and table[tdict['A_BC']][cdict['C']] > C_current:
        return (False, 'Not stable due to A_BC')
    
    # Check coalition {A}
    if table[tdict['A_B_C_']][cdict['A']] > A_current:
        return (False, 'Not stable due to A in A_B_C_')
    
    # Check coalition {C}
    if table[tdict['A_B_C_']][cdict['C']] > C_current:
        return (False, 'Not stable due to C in A_B_C_')
    
    return (True, 'Core stable')

def test_A_BC_stability(table):
    A_current = table[tdict['A_BC']][cdict['A']]
    B_current = table[tdict['A_BC']][cdict['B']]
    C_current = table[tdict['A_BC']][cdict['C']]
    
    # Check coalition {A,B,C}
    if table[tdict['ABC']][cdict['A']] > A_current and table[tdict['ABC']][cdict['B']] > B_current and table[tdict['ABC']][cdict['C']] > C_current:
        return (False, 'Not stable due to ABC')
    
    # Check coalition {A,B}
    if table[tdict['AB_C']][cdict['A']] > A_current and table[tdict['AB_C']][cdict['B']] > B_current:
        return (False, 'Not stable due to AB_C')
    
    # Check coalition {A,C}
    if table[tdict['AC_B']][cdict['A']] > A_current and table[tdict['AC_B']][cdict['C']] > C_current:
        return (False, 'Not stable due to AC_B')
    
    # Check coalition {A}
    if table[tdict['A_B_C_']][cdict['A']] > A_current:
        return (False, 'Not stable due to A in A_B_C_')
    
    # Check coalition {B}
    if table[tdict['A_B_C_']][cdict['B']] > B_current:
        return (False, 'Not stable due to B in A_B_C_')
    
    # Check coalition {C}
    if table[tdict['A_B_C_']][cdict['C']] > C_current:
        return (False, 'Not stable due to C in A_B_C_')
    
    return (True, 'Core stable')

def test_A_B_C__stability(table):
    A_current = table[tdict['A_B_C_']][cdict['A']]
    B_current = table[tdict['A_B_C_']][cdict['B']]
    C_current = table[tdict['A_B_C_']][cdict['C']]
    
    # Check coalition {A,B,C}
    if table[tdict['ABC']][cdict['A']] > A_current and table[tdict['ABC']][cdict['B']] > B_current and table[tdict['ABC']][cdict['C']] > C_current:
        return (False, 'Not stable due to ABC')
    
    # Check coalition {A,B}
    if table[tdict['AB_C']][cdict['A']] > A_current and table[tdict['AB_C']][cdict['B']] > B_current:
        return (False, 'Not stable due to AB_C')
    
    # Check coalition {B,C}
    if table[tdict['AC_B']][cdict['B']] > B_current and table[tdict['AC_B']][cdict['C']] > C_current:
        return (False, 'Not stable due to AC_B')
            
    # Check coalition {A,C}
    if table[tdict['AC_B']][cdict['A']] > A_current and table[tdict['AC_B']][cdict['C']] > C_current:
        return (False, 'Not stable due to AC_B')

    return (True, 'Core stable')

def check_core_stability(final_table):
    text_name = ['ABC', 'AB_C', 'AC_B', 'A_BC', 'A_B_C_']
    result_dict = {}
    for partition in text_name:
        if partition == 'ABC':
            result_dict[partition] = f'stable ABC?: {test_ABC_stability(final_table)}'
        elif partition == 'AB_C':
            result_dict[partition] = f'stable AB_C?: {test_AB_C_stability(final_table)}'
        elif partition == 'AC_B':
            result_dict[partition] = f'stable AC_B?: {test_AC_B_stability(final_table)}'
        elif partition == 'A_BC':
            result_dict[partition] = f'stable A_BC?: {test_A_BC_stability(final_table)}'
        elif partition == 'A_B_C_':
            result_dict[partition] = f'stable A_B_C_?: {test_A_B_C__stability(final_table)}'
        else:
            result_dict[partition] = 'Error: Invalid partition name'
    return result_dict

def test_AB_C_individually_stable(table):
    A_current = table[tdict['ABC']][cdict['A']]
    B_current = table[tdict['ABC']][cdict['B']]
    C_current = table[tdict['ABC']][cdict['C']]
    
    # Rule 1
    if (table['ABC'][cdict['A']] >= A_current and
        table['ABC'][cdict['B']] >= B_current and
        table['ABC'][cdict['C']] > C_current):
        return False
    
    # Rule 2
    if (table['AC_B'][cdict['A']] > A_current and
        table['AC_B'][cdict['C']] >= C_current):
        return False
    
    # Rule 3
    if (table['A_BC'][cdict['B']] > B_current and
        table['A_BC'][cdict['C']] >= C_current):
        return False
    
    return True

def test_AB_C_individually_stable(table):
    A_current = table[tdict['ABC']][cdict['A']]
    B_current = table[tdict['ABC']][cdict['B']]
    C_current = table[tdict['ABC']][cdict['C']]
    
    # Rule 1
    if (table['ABC'][cdict['A']] >= A_current and
        table['ABC'][cdict['B']] >= B_current and
        table['ABC'][cdict['C']] > C_current):
        return False
    
    # Rule 2
    if (table['AC_B'][cdict['A']] > A_current and
        table['AC_B'][cdict['C']] >= C_current):
        return False
    
    # Rule 3
    if (table['A_BC'][cdict['B']] > B_current and
        table['A_BC'][cdict['C']] >= C_current):
        return False
    
    return True

def createTableFromCoalition(coalition, theta_max, is_uniform=True, is_squared=True, mean=1, sd=1):
    print('justin: createTableFromCoalition() called')
    results_array = []
    accuracies_as_table = [coalition.ABC, coalition.AB_C, coalition.AC_B, coalition.A_BC, coalition.A_B_C_]

    for i, partition in enumerate(accuracies_as_table):
        p_new, profits, ordering, j = optimize(i, partition, mean, sd, theta_max, is_uniform, is_squared)
        results_array.append((p_new, profits, ordering, j))
    print('justin: before get_fine_table')
    reordered_profits, reordered_prices = get_final_table(results_array)
    print('justin: after get_fine_table')
    np.set_printoptions(precision=8, suppress=True)
    print(f'prices: {reordered_prices}\n profits: {reordered_profits}')
    # For no competition, check stability with pure accuracy of each model
    profit_stability_dict = check_core_stability(reordered_profits)
    accuracy_stability_dict = check_core_stability(accuracies_as_table)

    return accuracies_as_table, reordered_profits, reordered_prices, profit_stability_dict, accuracy_stability_dict

def calculate_equilibrium_price(A1, A2, A3, theta_max):
    p1_star = ((A1 - A2) * (3 * A1 + A2 - 4 * A3) * theta_max) / (6 * (A1 - A3))
    p2_star = ((A1 - A2) * (A2 - A3) * theta_max) / (3 * (A1 - A3))
    p3_star = ((A1 - A2) * (A2 - A3) * theta_max) / (6 * (A1 - A3))
    return p1_star, p2_star, p3_star

def calculate_equilibrium_profits(A1, A2, A3, theta_max):
    W1 = ((A1 - A2) * (3 * A1 + A2 - 4 * A3) ** 2 * theta_max) / (36 * (A1 - A3) ** 2)
    W2 = ((A1 - A2) * (A2 - A3) * theta_max) / (9 * (A1 - A3))
    W3 = ((A1 - A2) ** 2 * (A2 - A3) * theta_max) / (36 * (A1 - A3) ** 2)
    return W1, W2, W3

def model_degredation_best_response(A_max):
    """
    This function implements the best response algorithm for the model degradation game.

    Args:
      A_max: The maximum possible model performance for each client

    Returns:
      A: The best response strategy for each client.
    """

    # Initialize the strategy profile.
    A = np.copy(A_max)

    # Iterate until convergence.
    while True:
        # Create a copy of the strategy profile for comparison.
        A_old = np.copy(A)

        # Update the strategy profile.
        for n in range(len(A)):
            print(f"n: {n} | Complete strategy {A}")
            # Find the best response for client n.
            print(f"Player {n+1}: Current strategy = {A[n]}")
            if A[n] > np.max([A[i] for i in range(len(A)) if i != n]):
                # Client has best performing model
                print('best performing')
                A[n] = A_max[n]
                print(f"Player {n+1}: Updated strategy = {A[n]}")
            elif np.min([A[i] for i in range(len(A)) if i != n]) < A[n] < np.max([A[i] for i in range(len(A)) if i != n]):
                # Client has 2nd best performing model
                print(f'2nd best performing , left side: {round(np.max([A[i] for i in range(len(A)) if i != n]) - A[n], 3)}, right side: {round(A[n] - np.min([A[i] for i in range(len(A)) if i != n]), 3)}')
                if np.max([A[i] for i in range(len(A)) if i != n]) - A[n] >= A[n] - np.min([A[i] for i in range(len(A)) if i != n]):
                    A[n] = A_max[n]
                    print(f"Player {n+1}: Updated strategy = {A[n]}")
                else:
                    A[n] = 0
                    print(f"Player {n+1}: Updated strategy = {A[n]}")
            elif A[n] < np.min([A[i] for i in range(len(A)) if i != n]):
                print('worst performing')
                # Client has worst performing model
                if np.max([A[i] for i in range(len(A)) if i != n]) - np.min([A[i] for i in range(len(A)) if i != n]) >= np.min([A[i] for i in range(len(A)) if i != n]) - A[n]:
                    A[n] = 0
                    print(f"Player {n+1}: Updated strategy = {A[n]}")
                else:
                    A[n] = A_max[n]
                    print(f"Player {n+1}: Updated strategy = {A[n]}")

        # Check for convergence.
        if np.allclose(A, A_old):
            print(f'Final strategy {A}')
            break

    return A

if __name__ == '__main__':
    args = get_args()
    custom_array = []
    non_iid_array = []
    is_uniform = True
    is_squared = True
    theta_max = 10000
    mean = 5000
    sd = 234
    print('justin: running model degradation for ABC:', [x for (_, x) in quantity_coalition.AC_B])
    model_degredation_best_response([x for (_, x) in quantity_coalition.AC_B])
    if is_uniform:
        mean = 1
        sd = 1
    CUSTOM_ARRAY_PICKLE_NAME = f'custom_array_prices_{theta_max}_{mean}_{sd}'
    NON_IID_ARRAY_PICKLE_NAME = f'non_iid_array_prices_{theta_max}_{mean}_{sd}'
    if not os.path.exists(CUSTOM_ARRAY_PICKLE_NAME):
        print('----- Custom Quantity: A=1000, B=2000, C=8000 -----')
        for i, partition in enumerate([quantity_coalition.ABC, quantity_coalition.AB_C, quantity_coalition.AC_B, quantity_coalition.A_BC, quantity_coalition.A_B_C_]):
            p_new, profits, ordering, j = optimize(i, partition, mean, sd, theta_max, is_uniform, is_squared)
            custom_array.append((p_new, profits, ordering, j))
        with open(f'f{CUSTOM_ARRAY_PICKLE_NAME}.pickle', 'wb') as handle:
            pickle.dump((custom_array), handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('----- Non-iid Label Dirichlet: A=3491, B=3029, C=2480 -----')
        for i, partition in enumerate([dirichlet_coalition.ABC, dirichlet_coalition.AB_C, dirichlet_coalition.AC_B, dirichlet_coalition.A_BC, dirichlet_coalition.A_B_C_]):
            p_new, profits, ordering, j = optimize(i, partition, mean, sd, theta_max, is_uniform, is_squared)
            non_iid_array.append((p_new, profits, ordering, j))
        with open(f'f{NON_IID_ARRAY_PICKLE_NAME}.pickle', 'wb') as handle:
            pickle.dump((non_iid_array), handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(f'f{NON_IID_ARRAY_PICKLE_NAME}.pickle', 'rb') as handle:
            non_iid_array = pickle.load(handle)
        with open(f'f{CUSTOM_ARRAY_PICKLE_NAME}.pickle', 'rb') as handle:
            custom_array = pickle.load(handle)

    np.set_printoptions(precision=8, suppress=True)

    final_custom_table, _ = get_final_table(custom_array)
    final_non_iid_table, _ = get_final_table(non_iid_array)

    print(f'Final profit table for iid case:\n {final_custom_table}')
    print(f'Final profit table for non-iid case:\n {final_non_iid_table}')

    print('--- IID Quantity stability ---')
    custom_stability_dict = check_core_stability(final_custom_table)
    for key, value in custom_stability_dict.items():
        print(f"{key}: {value}")
    print('--- Non-IID Label stability ---')
    non_idd_stability_dict = check_core_stability(final_non_iid_table)
    for key, value in non_idd_stability_dict.items():
        print(f"{key}: {value}")
    # quantity_arrays = [ABC_Quantity, AB_C_Quantity, AC_B_Quantity, A_BC_Quantity, A_B_C_Quantity]
    # dirichlet_arrays = [ABC_Dirichlet, AB_C_Dirichlet, AC_B_Dirichlet, A_BC_Dirichlet, A_B_C_Dirichlet]

    print('--- IID Accuracy Testing ---')
    quantity_arrays = np.array([[x[1] for x in row] for row in quantity_arrays])
    for key, value in check_core_stability(quantity_arrays).items():
        print(f"{key}: {value}")
    dirichlet_arrays = np.array([[x[1] for x in row] for row in dirichlet_arrays])
    print('--- Non IID Accuracy Testing ---')
    for key, value in check_core_stability(dirichlet_arrays).items():
        print(f"{key}: {value}")

