#Code author: Yunru Zheng
#Date: December 3, 2025

import numpy as np
import math
from .model import get_optimal_sigma


def mean_estimator(N, V, rng):
    """
    Finds the mean of f(X) for X with standard normal distribution

    Args:
        N = number of
        V = function
        rng = random number
        output = estimated variance for the original sample
    """
    X_f   = rng.standard_normal(N)
    Y_f   = V(X_f)
    mean_f = np.mean(Y_f)         #old variable
    return mean_f

def mean_new_estimator(N, V, rng, sigma):
    """
    Finds the mean of V(X) for X with standard normal distribution
    E[V(X)] = int V(X)*f(X) = int f(X)/g(X)*V(X)*g(X) = int L(X)*V(X)*g(X)
    for L(X) = sigma*exp(-X^2*(1-1/sigma^2)/2)

    Args:
        N = number of
        V = function
        rng = random number
        sigma = chosen std for g
        output = estimated variance after importance sampling
    """
    X_g   = sigma * rng.standard_normal(N)
    L_g   = sigma * np.exp(-(X_g**2/2)*(1 - 1/sigma**2))
    Y_g   = V(X_g) * L_g        # new variable after importance sampling
    mean_g = np.mean(Y_g)
    return mean_g

def mean_sigma(N, V, rng, sigma):
    """
    Finds the mean of f(X) for X with normal distribution of standard
    deviation sigma

    Args:
        N = number of
        V = function
        rng = random number
        output = estimated variance for the original sample
    """
    X_g   = rng.standard_normal(N) * sigma
    Y_g   = V(X_g)
    mean_g = np.mean(Y_g)
    return mean_g

def var_estimator(N, f, rng):
    """
       N = number of
       f = function
       rng = random number
       output = estimated variance for the original sample
    """
    X_f   = rng.standard_normal(N)
    Y_f   = f(X_f)
    var_f = np.var(Y_f)         #old variable
    return var_f

def var_new_estimator(N, f, rng, sigma):
    """
       N = number of
       f = function
       rng = random number
       sigma = chosen std for g
       output = estimated variance after importance sampling
    """
    X_g   = sigma * rng.standard_normal(N)
    L_g   = sigma * np.exp(-(X_g**2/2)*(1 - 1/sigma**2))
    Y_g   = f(X_g) * L_g        # new variable after importance sampling
    var_g = np.var(Y_g)
    return var_g

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