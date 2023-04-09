import jax.numpy as jnp
from jax import grad
from scipy.optimize import minimize

theta_max = 1000
A = [0.8, 0.7, 0.9]
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
        res = minimize(W1Obj, x0=p, jac=dW_dp, method='BFGS')
        return res.x
    elif i == 1:
        dW_dp = grad(W2Obj)
        res = minimize(W2Obj, x0=p, jac=dW_dp, method='BFGS')
        return res.x
    elif i == 2:
        dW_dp = grad(W3Obj)
        res = minimize(W3Obj, x0=p, jac=dW_dp, method='BFGS')
        return res.x

p_init = [0, 0, 0]
p_new = p_init.copy()

while True:
    for i in range(3):
        p_new= update_price(float(i), p_new)
        print(f'i: {i}, pnew: {p_new}')
    if jnp.allclose(jnp.array(p_init), jnp.array(p_new), rtol=1e-3):
        break
    p_init = p_new.copy()

print(f"Optimal prices: {p_new}")
