from bestresponse import sigma, H

import numpy as np
import copy
import time
from scipy.optimize import basinhopping
import pandas as pd
from datetime import datetime
import logging
import os
import hashlib
import pickle

def setup_logger():
    # Hash of the current time to create unique experiment directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    hash_object = hashlib.md5(timestamp.encode())
    exp_dir = "experiment_" + hash_object.hexdigest()[:6] + "_" + timestamp
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # Setup logger
    logging.basicConfig(filename=os.path.join(exp_dir, 'experiment.log'), level=logging.INFO)

    return exp_dir

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
    SEARCH_DIFF = 1
    logging.info(f"Initial Price: {p}")
    price_history = [[] for _ in range(N)]
    quality_history = [[] for _ in range(N)]
    while True:
        p_new = p.copy()
        logging.info(f'start price: {p_new}')
        for n in range(N):
            best_result = None
            for quality in np.linspace(max(0, A[n]-SEARCH_DIFF), min(A_max[n], A[n]+SEARCH_DIFF), num=10, endpoint=True):
                A_temp = A.copy()
                A_temp[n] = quality
                result = basinhopping(lambda x: -W_n(n, np.hstack([p[:n], x, p[n+1:]]), A_temp, theta_max, mean, sd, is_squared), p[n], minimizer_kwargs={'method': 'BFGS'})
                if best_result is None or result.fun < best_result.fun:
                    best_result = result
                    best_quality = quality
            p_new[n] = best_result.x[0]
            A[n] = best_quality
            logging.info(f'Updated quality of product for client {n}: {A[n]}')
            price_history[n].append(p_new[n])
            quality_history[n].append(A[n]) 
        logging.info(f"New Price: {p_new} niter {niter}")
        if np.linalg.norm(p_new - p) < tol or niter >= 100:
            break
        niter += 1
        p = p_new.copy()
    return p, A

def run_competition(exp_dir):
    A = np.array([85.07, 85.18, 85.07, 85.41, 85.17, 85.71, 85.53, 85.08, 85.53, 85.68]) / 100
    p_ini = np.ones(10)
    theta_max = 1000000
    mean = 500000
    sd = 5000
    is_squared = True

    # Calculate the final prices
    p_final, A_final = price_competition(p_ini, A, theta_max, mean, sd, is_squared)
    logging.info(f"Final Prices: {p_final}")
    logging.info(f"Final Quality Levels: {A_final}")

    # Calculate the final market shares
    M_final = market_share(p_final, A_final, theta_max, mean, sd, is_squared)
    logging.info(f"Final Market Shares: {M_final}")

    for i in range(len(A_final)):
        logging.info(f"Client {i+1}:")
        logging.info(f"  Initial model performance score: {A[i]}")
        logging.info(f"  Final model performance score: {A_final[i]}")
        logging.info(f"  Initial price: {p_ini[i]}")
        logging.info(f"  Final price: {p_final[i]}")
        logging.info(f"  Final market share: {M_final[i]}")

def create_tables(exp_dir):
    global price_history, quality_history
    price_df = pd.DataFrame(price_history).T  # transpose to get clients as columns
    price_df.columns = [f"Client {i+1}" for i in range(price_df.shape[1])]
    price_df.index.name = "Round"

    quality_df = pd.DataFrame(quality_history).T  # transpose to get clients as columns
    quality_df.columns = [f"Client {i+1}" for i in range(quality_df.shape[1])]
    quality_df.index.name = "Round"

    # Pickle the dataframes
    with open(os.path.join(exp_dir, 'price_df.pkl'), 'wb') as f:
        pickle.dump(price_df, f)
    with open(os.path.join(exp_dir, 'quality_df.pkl'), 'wb') as f:
        pickle.dump(quality_df, f)

    return price_df, quality_df

exp_dir = setup_logger()

run_competition(exp_dir)
price_df, quality_df = create_tables(exp_dir)

logging.info("Price history:")
logging.info(price_df)
logging.info("\nQuality history:")
logging.info(quality_df)
