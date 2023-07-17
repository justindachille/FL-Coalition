from bestresponse import sigma, H

import numpy as np
import copy
import time
from scipy.optimize import basinhopping
import pandas as pd


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
    # time.sleep(.02)
    return market_share_result[n] * p[n]

price_history = []
quality_history = []

def price_competition(p_ini, A, theta_max, mean, sd, is_squared, tol=1e-5):
    global price_history, quality_history
    A_max = copy.deepcopy(A)
    N = len(p_ini)
    p = p_ini.copy()
    niter = 0
    print(f"Initial Price: {p}")
    price_history = [[] for _ in range(N)]
    quality_history = [[] for _ in range(N)]
    while True:
        p_new = p.copy()
        print(f'start price: {p_new}')
        for n in range(N):
            best_result = None
            for quality in np.linspace(max(0, A[n]-0.05), min(A_max[n], A[n]+0.05), num=10, endpoint=True):  # Modify the num parameter to adjust the granularity of the grid search
                A_temp = A.copy()
                A_temp[n] = quality
                result = basinhopping(lambda x: -W_n(n, np.hstack([p[:n], x, p[n+1:]]), A_temp, theta_max, mean, sd, is_squared), p[n], minimizer_kwargs={'method': 'BFGS'})
                if best_result is None or result.fun < best_result.fun:
                    # print(f'found new best price: {result.x[0]} with quality {quality}')
                    best_result = result
                    best_quality = quality
            p_new[n] = best_result.x[0]
            A[n] = best_quality
            print(f'Updated quality of product for client {n}: {A[n]}')
            price_history[n].append(p_new[n])
            quality_history[n].append(A[n]) 
        print(f"New Price: {p_new} niter {niter}")
        if np.linalg.norm(p_new - p) < tol or niter >= 450:
            break
        niter += 1
        p = p_new.copy()
    return p, A

def run_competition():
    A = np.array([85.07, 85.18, 85.07, 85.41, 85.17, 85.71, 85.53, 85.08, 85.53, 85.68]) / 100
    p_ini = np.ones(10)
    theta_max = 10000
    mean = 5000
    sd = 100
    is_squared = False

    # Calculate the final prices
    p_final, A_final = price_competition(p_ini, A, theta_max, mean, sd, is_squared)
    print(f"Final Prices: {p_final}")
    print(f"Final Quality Levels: {A_final}")

    # Calculate the final market shares
    M_final = market_share(p_final, A_final, theta_max, mean, sd, is_squared)
    print(f"Final Market Shares: {M_final}")

    for i in range(len(A_final)):
        print(f"Client {i+1}:")
        print(f"  Initial model performance score: {A[i]}")
        print(f"  Final model performance score: {A_final[i]}")
        print(f"  Initial price: {p_ini[i]}")
        print(f"  Final price: {p_final[i]}")
        print(f"  Final market share: {M_final[i]}")

def create_tables():
    global price_history, quality_history
    price_df = pd.DataFrame(price_history).T  # transpose to get clients as columns
    price_df.columns = [f"Client {i+1}" for i in range(price_df.shape[1])]
    price_df.index.name = "Round"

    quality_df = pd.DataFrame(quality_history).T  # transpose to get clients as columns
    quality_df.columns = [f"Client {i+1}" for i in range(quality_df.shape[1])]
    quality_df.index.name = "Round"

    return price_df, quality_df


run_competition()
price_df, quality_df = create_tables()

print("Price history:")
print(price_df)
print("\nQuality history:")
print(quality_df)

