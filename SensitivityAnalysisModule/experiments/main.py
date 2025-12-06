#Code author: Yunru Zheng
#Date: December 3, 2025

import numpy as np
import matplotlib.pyplot as plt
import math
from src.model import get_optimal_sigma, m_func, grad_m_func, h_func, grad_h_func
from src.optimize import run_sgd_optimization
from src.utils import print_optimization_process, print_results_table, calculate_gradient_variance

N_SAMPLES = 50000
LEARNING_RATE = 0.005
CLIP_THRESHOLD = 10000.0
K_MAX = 20
RNG = np.random.default_rng(17)
BATCH_SIZE = 50

# Goal 1: Checking the effect of learning rate on getting optimal sigma
# for X^2

learning_rates = [0.0005, 0.001, 0.005]

# Using the approach to differentiate the integrand directly and using
# X ~ N(0, sigma)

for lr in learning_rates:
    tl = f"\n Direct Gradient Descent using X ~ N(0, sigma) | batch size = 1 | learning_rate: {lr}"
    print(tl)
    print('-'*100)
    s_d_histx, l_d_histx = run_sgd_optimization(k=1,
                                                grad_func=grad_m_func,
                                                loss_func=m_func,
                                                batch_size=1,
                                                learning_rate=lr,
                                                is_reparam=False)
    # Printing Results
    print_optimization_process(s_d_histx, l_d_histx, step=2000)


# Using the approach to set X = sigma*Z and differentiate the integrand
# using Z ~ N(0, 1)

for lr in learning_rates:
    tl = f"\n Direct Gradient Descent using Z ~ N(0, 1) | batch size = 1 | learning_rate: {lr}"
    print(tl)
    print('-'*100)
    s_d_histz, l_d_histz = run_sgd_optimization(k=1,
                                                grad_func=grad_h_func,
                                                loss_func=h_func,
                                                batch_size=1,
                                                learning_rate=lr,
                                                is_reparam=True)
    # Printing Results
    print_optimization_process(s_d_histz, l_d_histz, step=2000)

# Goal 2: I'm going to use Batch Gradient Descent to see if the results would be more stable.
# After looking at the results from step 1, I decided to use step size 0.005.

# Using the to differentiate the integrand directly and using
# X ~ N(0, sigma)

tl1 = f"\nBatch Gradient Descent using X ~ N(0, sigma) | batch size = {BATCH_SIZE} |"
tl1 = tl1 + f" learning_rate: {LEARNING_RATE}"
print(tl1)
print('-'*100)

s_b_histx, l_b_histx = run_sgd_optimization(k=1,
                                            grad_func=grad_m_func,
                                            loss_func=m_func,
                                            batch_size=BATCH_SIZE,
                                            learning_rate=LEARNING_RATE,
                                            is_reparam=False)
print_optimization_process(s_b_histx, l_b_histx, step=50)


# Using the approach to set X = sigma*Z and differentiate the integrand
# using Z ~ N(0, 1)

tl2 = f"\nBatch Gradient Descent using Z ~ N(0, 1) | batch size = {BATCH_SIZE} |"
tl2 = tl2 + f" learning_rate: {LEARNING_RATE}"
print(tl2)
print('-'*100)

s_b_histz, l_b_histz = run_sgd_optimization(k=1,
                                            grad_func=grad_h_func,
                                            loss_func=h_func,
                                            batch_size=BATCH_SIZE,
                                            learning_rate=LEARNING_RATE,
                                            is_reparam=True)
print_optimization_process(s_b_histz, l_b_histz, step=50)


# Goal 3: I'm trying to show if I can find reliable optimal sigma value as the moment gets larger

results_x = []
results_z = []
for k in range(1, K_MAX+1):
    sigma_opt = get_optimal_sigma(k)

    # Run X-Method
    s_hist_x, _ = run_sgd_optimization(k=k,
                                       grad_func=grad_m_func,
                                       loss_func=m_func,
                                       batch_size=BATCH_SIZE,
                                       learning_rate=0.0005,
                                       is_reparam=False)
    sigma_final_x = np.mean(s_hist_x[-50:])
    err_x = abs(sigma_final_x - sigma_opt)
    results_x.append((k, sigma_final_x, sigma_opt, err_x))

    # Run Z-Method
    s_hist_z, _ = run_sgd_optimization(k=k,
                                       grad_func=grad_m_func,
                                       loss_func=m_func,
                                       batch_size=BATCH_SIZE,
                                       learning_rate=0.0005,
                                       is_reparam=False)
    sigma_final_z = np.mean(s_hist_z[-50:])
    err_z = abs(sigma_final_z - sigma_opt)
    results_z.append((k, sigma_final_z, sigma_opt, err_z))

    # Print results tables
print_results_table("FINAL RESULTS: X-Method (N(0, sigma))", results_x)
print_results_table("FINAL RESULTS: Z-Method (N(0, 1))", results_z)

# Goal 4: Diagnostics (Gradient Variance)

print("\n" + "="*60)
print("GOAL 4: Diagnostic - Gradient Noise Comparison")
print("="*60)

diagnostic_samples = RNG.standard_normal(N_SAMPLES)

print(f"{'k':<5} | {'Var(Grad X)':<20} | {'Var(Grad Z)':<20} | {'Ratio (VarX/VarZ)':<15}")
print("-" * 65)

for k in range(1, K_MAX+1):
    # Gradient variance for X-Method
    var_x = calculate_gradient_variance(k, diagnostic_samples,
                                        m_func, grad_m_func,
                                        is_reparam=False)
    # Gradient variance for Z-Method
    var_z = calculate_gradient_variance(k, diagnostic_samples,
                                        h_func, grad_h_func,
                                        is_reparam=True)
    # Calculate the ratio
    if var_z > 0 and var_x != np.inf:
        ratio = f"{var_x / var_z: .1e}"
    else:
        ratio = "N/A"

    # Print results
    print(f"{k:<5} | {var_x:<20.2e} | {var_z:<20.2e} | {ratio:<15}")