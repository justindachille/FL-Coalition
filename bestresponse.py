import argparse
import os

# import jax.numpy as jnp
# from jax import grad
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import truncnorm

from scipy.optimize import basinhopping
import dill as pickle

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
A_B_C_Quantity = [SOLO_QUANTITY_A, SOLO_QUANTITY_B, SOLO_QUANTITY_C]
AB_C_Quantity = [(0, .7976), (1, .8007), SOLO_QUANTITY_C]
A_BC_Quantity = [SOLO_QUANTITY_A, (1, .8732), (2, .8762)]
AC_B_Quantity = [(0, 0.8550), SOLO_QUANTITY_B, (2, .8608)]
ABC_Quantity = [(0, .8803), (1, .8821), (2, .8817)]

# Non-iid
SOLO_DIRICHLET_A = (0, .6085)
SOLO_DIRICHLET_B = (1, .6394)
SOLO_DIRICHLET_C = (2, .5932)
A_B_C_Dirichlet = [SOLO_DIRICHLET_A, SOLO_DIRICHLET_B, SOLO_DIRICHLET_C]
AB_C_Dirichlet = [(0, .8440), (1, .8453), SOLO_DIRICHLET_C]
A_BC_Dirichlet = [SOLO_DIRICHLET_A, (1, .8347), (2, .8320)]
AC_B_Dirichlet = [(0, 0.8301), SOLO_DIRICHLET_B, (2, .8359)]
ABC_Dirichlet = [(0, .8681), (1, .8668), (2, .8655)]

C_pri = [0.5, 0.3, 0.1]

def sigma(m, n, p, A):
    return (p[m] - p[n]) / (A[m] - A[n]) # Change to A^2 - A^2

def N(theta, mean, sd, is_uniform=False):
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
# pdf_vals = [N(theta, mean, sd, is_uniform=True) for theta in theta_vals]

# # create the plot
# fig, ax = plt.subplots()
# ax.plot(theta_vals, pdf_vals, label='Truncated normal distribution')
# ax.set_xlabel('Theta')
# ax.set_ylabel('PDF')
# ax.legend()
# plt.show()

def W0(p, A):
    # print(p[0], (1 - N(max(sigma(0, 1, p, A), sigma(0, 2, p, A)), mean, sd)))
    return p[0] * (1 - N(max(sigma(0, 1, p, A), sigma(0, 2, p, A)), mean, sd, is_uniform))# - C_pri[0]

def W1(p, A):
    return p[1] * N(sigma(0, 1, p, A) - sigma(1, 2, p, A), mean, sd, is_uniform)# - C_pri[1]

def W2(p, A):
    return p[2] * N(min(sigma(1, 2, p, A), sigma(0, 2, p, A)), mean, sd, is_uniform) #- C_pri[2]

def W0Obj(p, A):
    return -W0(p, A)

def W1Obj(p, A):
    return -W1(p, A)

def W2Obj(p, A):
    return -W2(p, A)

def update_price(i, p, A):
    if i == 0:
        res = basinhopping(lambda x: W0Obj([x, p[1], p[2]], A), x0=p[0], minimizer_kwargs={'method': 'BFGS'})
        return [res.x[0], p[1], p[2]]
    elif i == 1:
        res = basinhopping(lambda x: W1Obj([p[0], x, p[2]], A), x0=p[1], minimizer_kwargs={'method': 'BFGS'})
        return [p[0], res.x[0], p[2]]
    elif i == 2:
        res = basinhopping(lambda x: W2Obj([p[0], p[1], x], A), x0=p[2], minimizer_kwargs={'method': 'BFGS'})
        return [p[0], p[1], res.x[0]]

text_name = ['ABC', 'AB_C', 'AC_B', 'A_BC', 'A_B_C_']
quantity_arrays = [ABC_Quantity, AB_C_Quantity, AC_B_Quantity, A_BC_Quantity, A_B_C_Quantity]
dirichlet_arrays = [ABC_Dirichlet, AB_C_Dirichlet, AC_B_Dirichlet, A_BC_Dirichlet, A_B_C_Dirichlet]

def get_profit(i, price, scores):
    if i == 0:
        return W0(price, scores)
    elif i == 1:
        return W1(price, scores)
    elif i == 2:
        return W2(price, scores)

def optimize(j, partition):
    print(partition)
    partition = sorted(partition, key=lambda x: x[1], reverse=True)
    print(partition)
    p_init = [5] * 3
    p_new = p_init.copy()
    ordering = []
    scores = []
    for ord, score in partition:
        ordering += [ord]
        scores += [score]

    while True:
        for i in range(3):
            p_new = update_price(float(i), p_new, scores)
            # print(f'i: {i}, pnew: {p_new}')
        if np.allclose(np.array(p_init), np.array(p_new), rtol=1e-6):
            break
        p_init = p_new.copy()

    print(f'Optimal prices: {p_new} with order {ordering} for partition {text_name[j]}')
    profits = []
    for i in range(3):
        profits.append(get_profit(i, p_new, scores))
    print(f'Profits: {profits}')
    return p_new, profits, ordering, j
    
if __name__ == '__main__':
    args = get_args()
    custom_array = []
    non_iid_array = []
    is_uniform = True
    theta_max = 10000
    mean = 5000
    sd = 234
    if is_uniform:
        mean = 1
        sd = 1
    CUSTOM_ARRAY_PICKLE_NAME = f'custom_array_prices_{theta_max}_{mean}_{sd}'
    NON_IID_ARRAY_PICKLE_NAME = f'non_iid_array_prices_{theta_max}_{mean}_{sd}'
    if not os.path.exists(CUSTOM_ARRAY_PICKLE_NAME):
        print('----- Custom Quantity: A=1000, B=2000, C=8000 -----')
        for i, partition in enumerate(quantity_arrays):
            p_new, profits, ordering, j = optimize(i, partition)
            custom_array.append((p_new, profits, ordering, j))
        with open(f'f{CUSTOM_ARRAY_PICKLE_NAME}.pickle', 'wb') as handle:
            pickle.dump((custom_array), handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('----- Non-iid Label Dirichlet: A=3491, B=3029, C=2480 -----')
        for i, partition in enumerate(dirichlet_arrays):
            p_new, profits, ordering, j = optimize(i, partition)
            non_iid_array.append((p_new, profits, ordering, j))
        with open(f'f{NON_IID_ARRAY_PICKLE_NAME}.pickle', 'wb') as handle:
            pickle.dump((non_iid_array), handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(f'f{NON_IID_ARRAY_PICKLE_NAME}.pickle', 'rb') as handle:
            non_iid_array = pickle.load(handle)
        with open(f'f{CUSTOM_ARRAY_PICKLE_NAME}.pickle', 'rb') as handle:
            custom_array = pickle.load(handle)

    # print('custom_array:', custom_array)
    # print('nonidd_array:', non_iid_array)

    def get_final_table(custom_array):
        table = []
        for i, partition in enumerate(text_name):
            prices, profits, ordering, _ = custom_array[i]
            profits = np.array(profits)
            print(f'profits before: {profits} ordering: {ordering}')
            profits_by_ordering = [0] * 3
            for i, order in enumerate(ordering):
                profits_by_ordering[order] = profits[i]
            print(f'profits after: {profits_by_ordering}')
            print(f'prices before: {prices} ordering: {ordering}')
            prices_by_ordering = [0] * 3
            for i, order in enumerate(ordering):
                prices_by_ordering[order] = prices[i]
            print(f'prices after: {prices_by_ordering}')
            table.append(profits_by_ordering)

        return table
    np.set_printoptions(precision=8, suppress=True)
    final_custom_table = np.array(get_final_table(custom_array))
    final_non_iid_table = np.array(get_final_table(non_iid_array))
    # text_name = ['ABC', 'AB_C', 'AC_B', 'A_BC', 'A_B_C_']

    print(f'Final profit table for iid case:\n {final_custom_table}')
    print(f'Final profit table for non-iid case:\n {final_non_iid_table}')
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

    def check_stability(final_table):
        for i, partition in enumerate(text_name):
            if partition == 'ABC':
                print('stable ABC?:', test_ABC_stability(final_table))
            elif partition == 'AB_C':
                print('stable AB_C?:', test_AB_C_stability(final_table))
            elif partition == 'AC_B':
                print('stable AC_B?:', test_AC_B_stability(final_table))
            elif partition == 'A_BC':
                print('stable A_BC?:', test_A_BC_stability(final_table))
            elif partition == 'A_B_C_':
                print('stable A_B_C_?:', test_A_B_C__stability(final_table))
            else:
                print('Error: Invalid partition name')

    print('--- IID Quantity stability ---')
    check_stability(final_custom_table)
    print('--- Non-IID Label stability ---')
    check_stability(final_non_iid_table)
    # quantity_arrays = [ABC_Quantity, AB_C_Quantity, AC_B_Quantity, A_BC_Quantity, A_B_C_Quantity]
    # dirichlet_arrays = [ABC_Dirichlet, AB_C_Dirichlet, AC_B_Dirichlet, A_BC_Dirichlet, A_B_C_Dirichlet]

    print('--- IID Accuracy Testing ---')
    quantity_arrays = np.array([[x[1] for x in row] for row in quantity_arrays])
    check_stability(quantity_arrays)

    dirichlet_arrays = np.array([[x[1] for x in row] for row in dirichlet_arrays])
    print('--- NON IID Accuracy Testing ---')
    check_stability(dirichlet_arrays)
    

