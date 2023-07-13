from bestresponse import sigma, H

import numpy as np
from scipy.stats import truncnorm
from scipy.optimize import basinhopping
import warnings


def market_share(p, A, theta_max, mean, sd, is_squared):
    N = len(p)
    sigma_values = np.zeros((N, N))
    for m in range(N):
        for n in range(m+1, N):
            sigma_values[m, n] = sigma(m, n, p, A, is_squared)
            sigma_values[n, m] = sigma(n, m, p, A, is_squared)

    M = np.zeros(N)
    M[0] = H(theta_max, mean, sd, theta_max) - H(np.max(sigma_values[0, 1:]), mean, sd, theta_max)
    for i in range(1, N-1):
        M[i] = H(sigma_values[i-1, i] - np.max([*sigma_values[i, i+1:], 0]), mean, sd, theta_max)
    M[-1] = H(np.min(np.append(sigma_values[:-1, -1], theta_max)), mean, sd, theta_max) - H(0, mean, sd, theta_max)
    return M

def W_n(n, p, A, theta_max, mean, sd, is_squared):
    return market_share(p, A, theta_max, mean, sd, is_squared)[n] * p[n]

def price_competition(p_ini, A, theta_max, mean, sd, is_squared, tol=1e-5):
    N = len(p_ini)
    p = p_ini.copy()
    while True:
        p_new = p.copy()
        for n in range(N):
            result = basinhopping(lambda x: -W_n(n, np.hstack([p[:n], x, p[n+1:]]), A, theta_max, mean, sd, is_squared), p[n], minimizer_kwargs={'method': 'BFGS'}, niter=100)
            p_new[n] = result.x[0]
        if np.linalg.norm(p_new - p) < tol:
            break
        p = p_new
    return p

def run_competition():
    A = np.array([85.07, 85.18, 85.07, 85.41, 85.17, 85.71, 85.53, 85.08, 85.53, 85.68]) / 100
    p_ini = np.ones(10)*10  # initial prices; assuming 1 for each client
    theta_max = 10000
    mean = 5000
    sd = 100
    is_squared = False

    # Calculate the final prices
    p_final = price_competition(p_ini, A, theta_max, mean, sd, is_squared)

    # Calculate the final market shares
    M_final = market_share(p_final, A, theta_max, mean, sd, is_squared)

    for i in range(len(A)):
        print(f"Client {i+1}:")
        print(f"  Model performance score: {A[i]}")
        print(f"  Initial price: {p_ini[i]}")
        print(f"  Final price: {p_final[i]}")
        print(f"  Final market share: {M_final[i]}")
  
run_competition()