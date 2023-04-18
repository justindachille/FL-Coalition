import jax.numpy as jnp
from jax import grad
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import basinhopping

theta_max = 10
A = [0.9, 0.8, 0.7]
# Custom Quantity
SOLO_QUANTITY_A = (0, .5004)
SOLO_QUANTITY_B = (1, .6116)
SOLO_QUANTITY_C = (2, .7091)
A_B_C_Quantity = [SOLO_QUANTITY_A, SOLO_QUANTITY_B, SOLO_QUANTITY_C]
AB_C_Quantity = [(0, .7976), (1, .8007), SOLO_QUANTITY_C]
A_BC_Quantity = [SOLO_QUANTITY_A, (1, .8762), (2, .8732)]
AC_B_Quantity = [(0, 0.8550), SOLO_QUANTITY_B, (2, .8608)]
ABC_Quantity = [(0, .8810), (1, .8821), (2, .8817)]

C_pri = [0.5, 0.3, 0.1]

def sigma(m, n, p, A):
    return (p[m] - p[n]) / (A[m] - A[n])

def H(theta):
    if theta < 0:
        return 0
    elif theta <= theta_max:
        return theta / theta_max
    else:
        return 1

def W1(p, A):
    return p[0] * (1 - H(max(sigma(0, 1, p, A), sigma(0, 2, p, A))))# - C_pri[0]

def W2(p, A):
    return p[1] * H(sigma(0, 1, p, A) - sigma(1, 2, p, A))# - C_pri[1]

def W3(p, A):
    return p[2] * H(min(sigma(1, 2, p, A), sigma(0, 2, p, A))) #- C_pri[2]

def W1Obj(p, A):
    return -W1(p, A)

def W2Obj(p, A):
    return -W2(p, A)

def W3Obj(p, A):
    return -W3(p, A)

def update_price(i, p, A):
    if i == 0:
        res = basinhopping(lambda x: W1Obj([x, p[1], p[2]], A), x0=p[0], minimizer_kwargs={'method': 'BFGS'})
        return [res.x[0], p[1], p[2]]
    elif i == 1:
        res = basinhopping(lambda x: W2Obj([p[0], x, p[2]], A), x0=p[1], minimizer_kwargs={'method': 'BFGS'})
        return [p[0], res.x[0], p[2]]
    elif i == 2:
        res = basinhopping(lambda x: W3Obj([p[0], p[1], x], A), x0=p[2], minimizer_kwargs={'method': 'BFGS'})
        return [p[0], p[1], res.x[0]]

quantity_arrays = [A_B_C_Quantity, AB_C_Quantity, A_BC_Quantity, AC_B_Quantity, ABC_Quantity]

for partition in quantity_arrays:
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
        if jnp.allclose(jnp.array(p_init), jnp.array(p_new), rtol=1e-3):
            break
        p_init = p_new.copy()

    print(f"Optimal prices: {p_new} with order {ordering}")
    break
