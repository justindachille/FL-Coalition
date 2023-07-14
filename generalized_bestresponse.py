from bestresponse import sigma, H

import numpy as np
from scipy.optimize import basinhopping


def market_share(p, A, theta_max, mean, sd, is_squared):
    sorted_indices = np.argsort(A)[::-1]
    # print(f"A (Performance in original order): {A}")
    # print(f"p (Strategy in original order): {p}")
    A = A[sorted_indices]
    p = p[sorted_indices]
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

    reverse_indices = np.argsort(sorted_indices)
    M_original = M[reverse_indices]

    # print(f"M (Market shares in original order): {M_original}")

    return M_original

def W_n(n, p, A, theta_max, mean, sd, is_squared):
    market_share_result = market_share(p, A, theta_max, mean, sd, is_squared)
    # print(f'p: {p} A: {A}')
    # print(f"Market Share Result for n={n}: {market_share_result}")
    return market_share_result[n] * p[n]

def price_competition(p_ini, A, theta_max, mean, sd, is_squared, tol=1e-5):
    N = len(p_ini)
    p = p_ini.copy()
    print(f"Initial Price: {p}")
    while True:
        p_new = p.copy()
        print(f'start price: {p_new}')
        for n in range(N):
            result = basinhopping(lambda x: -W_n(n, np.hstack([p[:n], x, p[n+1:]]), A, theta_max, mean, sd, is_squared), p[n], minimizer_kwargs={'method': 'BFGS'})
            print(f'found new price: {result.x[0]}')
            p_new[n] = result.x[0]
        print(f"New Price: {p_new}")
        if np.linalg.norm(p_new - p) < tol:
            break
        p = p_new.copy()
    return p

def run_competition():
    A = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]) / 100
    p_ini = np.ones(10) * 50
    theta_max = 10000
    mean = 5000
    sd = 100
    is_squared = False

    # Calculate the final prices
    p_final = price_competition(p_ini, A, theta_max, mean, sd, is_squared)
    print(f"Final Prices: {p_final}")

    # Calculate the final market shares
    M_final = market_share(p_final, A, theta_max, mean, sd, is_squared)
    print(f"Final Market Shares: {M_final}")

    for i in range(len(A)):
        print(f"Client {i+1}:")
        print(f"  Model performance score: {A[i]}")
        print(f"  Initial price: {p_ini[i]}")
        print(f"  Final price: {p_final[i]}")
        print(f"  Final market share: {M_final[i]}")
  
run_competition()