
#Code author: Yunru Zheng
#Date: November 14, 2025

import numpy as np
import matplotlib.pyplot as plt
import math

N_SAMPLES = 50000
LEARNING_RATE = 0.005
CLIP_THRESHOLD = 10000.0
K_MAX = 20
RNG = np.random.default_rng(17)
BATCH_SIZE = 50

# We are doing sensitivity analysis on the std of our importance function g(x) = N(0,sigma).
# We will find the gradient of the variance with respect to sigma, and use statistical
# approximation to find the optimal sigma value by gradient descent.

K_PARAM = 1

def get_optimal_sigma(k):
    """
        Returns the analytical optimal sigma for a given k.
    """
    return np.sqrt(2*k + 1)

# Method 1: The "X" Method (LRM)
def m_func(X, sigma, k=K_PARAM):
    """
        x = N(0, sigma)
        sigma = the std for g, and we are trying various values to find the optimal
        one that minimizes the variance after importance sampling
        k = a variable that determines the moment: n = 2k, <X^n> = <X^{2k}> is an even moment
    """
    exponent = -np.power(X,2) * (1-1/np.power(sigma,2))
    term1 = np.power(sigma,2) * np.exp(np.clip(exponent, -np.inf, 700))
    term2 = np.power(X, 4*k)
    return term1 * term2

def grad_m_func(x, sigma, m_val, k=K_PARAM):
    """
        x = N(0, sigma)
        sigma = the std for g, and we are trying various values to find the optimal
        one that minimizes the variance after importance sampling
        m_func = the m function and we are finding the gradient as it multiplying some value
        k = a variable that determines the moment: n = 2k, <X^n> = <X^{2k}> is an even moment
    """
    term1 = (np.power(sigma, 2) - np.power(x,2)) / np.power(sigma, 3)
    return term1 * m_val

# Method 2: The "Z" Method (Reparametrization)
def h_func(Z, sigma, k=K_PARAM):
    """
        Z = standard normal distributed variable
        sigma = the std for g, and we are trying various values to find the optimal
        one that minimizes the variance after importance sampling
        k = a variable that determines the moment: n = 2k, <X^n> = <X^{2k}> is an even moment
    """
    term1 = np.power(sigma, 4*k + 2)
    term2 = np.power(Z, 4*k)
    exponent = np.power(Z, 2) * (1 - np.power(sigma, 2))
    term3 = np.exp(np.clip(exponent, -np.inf, 700))

    return term1 * term2 * term3

def grad_h_func(Z, sigma, h_func, k=K_PARAM):
    """
        Z = standard normal distributed variable
        sigma = the std for g, and we are trying various values to find the optimal
        one that minimizes the variance after importance sampling
        h_func = the h function and we are finding the gradient as it multiplying some value
        k = a variable that determines the moment: n = 2k, <X^n> = <X^{2k}> is an even moment
    """

    term1 = (4*k + 2) / sigma
    term2 = 2 * sigma * np.power(Z, 2)
    gradient = (term1 - term2) * h_func
    return gradient

# HELPER FUNCTION: 
# is_reparam = True corresponds to Z method, and False for X method

def run_sgd_optimization(k, grad_func, loss_func, batch_size, learning_rate, is_reparam=True):
    sigma_opt = get_optimal_sigma(k)    # the optimal value sigma = \sqrt{2k+1}
    sigma_current = np.sqrt(2 * k)      # start slightly off from the optimal sigma value

    sigma_history = []
    loss_history = []

    # Calculate how many steps to make based on batch size
    # Batch=1   -> 50,000 steps
    # Batch=50  ->  1,000 steps

    num_iternations = N_SAMPLES // batch_size

    for i in range(num_iternations):
        # 1. Generate Batch
        raw_samples = RNG.standard_normal(batch_size)

        if is_reparam:
            #Z Method: Z ~ N(0, 1)
            batch_input = raw_samples
        else:
            #X Method: X ~ N(0, sigma)
            batch_input = raw_samples * sigma_current

        # 2. Calculate Gradient
        loss_vals = loss_func(batch_input, sigma_current, k)
        grad_vals = grad_func(batch_input, sigma_current, loss_vals, k)

        # 3. Cleaning and Clipping
        valid_grads = grad_vals[~(np.isinf(grad_vals) | np.isnan(grad_vals))]

        if not valid_grads.any(): # Check if the batch was empty after cleaning
            sigma_history.append(sigma_current)
            loss_history.append(loss_history[-1] if loss_history else 0)
            continue

        avg_grad = np.mean(valid_grads)
        avg_grad = np.clip(avg_grad, -CLIP_THRESHOLD, CLIP_THRESHOLD)

        # 4. Update
        sigma_current = sigma_current - learning_rate * avg_grad

        if sigma_current < 0.75:
            sigma_current = 0.75

        sigma_history.append(sigma_current)
        loss_history.append(np.mean(loss_vals))
    return sigma_history, loss_history

def print_optimization_process(sigma_hist, loss_hist, step=1000):
    """
    Helper function that prints out the results of the training history
    """
    for i in range(0, len(sigma_hist), step):
        if i == 0: continue
        avg_loss = np.mean(loss_hist[max(0, i-step):i])
        print(f"Iteration {i:>5}: sigma = {sigma_hist[i]:.4f}, Est.Loss (avg) = {avg_loss:.4f}")

def print_results_table(title, results):
    print("\n" + "="*60)
    print(f"--- {title} ---")
    print("="*60)
    print(f"{'n (X^2n)':<10} | {'Sigma':<15} | {'Optimal':<15} | {'Error':<15}")
    print("-"*60)
    for res in results:
        k, found, opt, err = res
        print(f"{k:<10} | {found:<15.4f} | {opt:<15.4f} | {err:<15.4f}")

def calculate_gradient_variance(k, raw_samples, loss_func, grad_func, is_reparam=True):
    """
    Calculates the variance of the gradient estimator at the optimal sigma.
    Returns the variance (scalar).
    """
    sigma_opt = get_optimal_sigma(k)

    # 1. Decides the method
    if is_reparam:
        # Z-Method: Input is Z ~ N(0, 1)
        model_input = raw_samples
    else:
        # X-Method: Input is X ~ N(0, sigma)
        model_input = raw_samples * sigma_opt

    # 2. Calculate Gradients
    loss_vals = loss_func(model_input, sigma_opt, k)
    grad_vals = grad_func(model_input, sigma_opt, loss_vals, k)

    # 3. Clean Inf/NaN
    valid_grads = grad_vals[~(np.isinf(grad_vals) | np.isnan(grad_vals))]

    # 4. Calculate Variance
    if valid_grads.size > 0:
        return np.var(valid_grads)
    else:
        return np.inf # Represent exploded variance as infinity

#-------  Main program -----------------------

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