import jax.numpy as jnp
from jax import grad
import matplotlib.pyplot as plt
from scipy.optimize import minimize

theta_max = 10
A = [0.9, 0.8, 0.7]
C_pri = [0.2, 0.3, 0.1]

def sigma(m, n, p):
    return (p[m] - p[n]) / (A[m] - A[n])

def H(theta):
    if theta < 0:
        return 0
    elif theta <= theta_max:
        return theta / theta_max
    else:
        return 1

def W1(p):
    return p[0] * (1 - H(max(sigma(0, 1, p), sigma(0, 2, p)))) - C_pri[0]

def W2(p):
    return p[1] * H(sigma(0, 1, p) - sigma(1, 2, p)) - C_pri[1]

def W3(p):
    return p[2] * H(min(sigma(1, 2, p), sigma(0, 2, p))) - C_pri[2]

def W1Obj(p):
    return -W1(p)

def W2Obj(p):
    return -W2(p)

def W3Obj(p):
    return -W3(p)

def update_price(i, p):
    if i == 0:
        dW_dp = grad(W1Obj)
        res = minimize(lambda x: W1([x, p[1], p[2]]), jac=dW_dp, x0=p[0], method='Powell')
        print(res)
        return [res.x[0], p[1], p[2]]
    elif i == 1:
        dW_dp = grad(W2Obj)
        res = minimize(lambda x: W2([p[0], x, p[2]]), jac=dW_dp, x0=p[1], method='Powell')
        print(res)
        return [p[0], res.x[0], p[2]]
    elif i == 2:
        dW_dp = grad(W3Obj)
        res = minimize(lambda x: W3([p[0], p[1], x]), jac=dW_dp, x0=p[2], method='Powell')
        print(res)
        return [p[0], p[1], res.x[0]]

p_init = [0.5, 0.5, 0.5]
p_new = p_init.copy()

while True:
    # test_values = [[p1, p2, p3] for p1 in range(1, 11, 5) for p2 in range(1, 11, 5) for p3 in range(1, 11, 5)]

    # for p in test_values:
    #     dW_dp = grad(W1Obj, allow_int=True)
    #     w1_value = dW_dp(p)
    #     print(f"p = {p}, W1 = {w1_value}")
    # create 100 evenly spaced values from 0 to 10

    ctr = 0
    for i in range(3):
        p_new = update_price(float(i), p_new)
        print(f'i: {i}, pnew: {p_new}')
        ctr += 1
    if jnp.allclose(jnp.array(p_init), jnp.array(p_new), rtol=1e-3) or ctr > 100:
        break
    p_init = p_new.copy()

print(f"Optimal prices: {p_new}")
