#Code author: Yunru Zheng
#Date: November 3, 2025

import numpy as np
import matplotlib.pyplot as plt
import math


# We are doing sensitivity analysis on the std of our importance function g(x) = N(0,sigma).
# We will find the gradient of the variance with respect to sigma, and use statistical
# approximation to find the optimal sigma value by gradient descent.

K_PARAM = 1

def h_func(epsilon, sigma, k=K_PARAM):
    """
        epsilon = standard normal distributed variable
        sigma = the std for g, and we are trying various values to find the optimal
        one that minimizes the variance after importance sampling
        k = a variable that determines the moment: n = 2k, <X^n> = <X^{2k}> is an even moment
    """
    term1 = np.power(sigma, 4*k + 2)
    term2 = np.power(epsilon, 4*k)
    exponent = np.power(epsilon, 2) * (1 - np.power(sigma, 2))
    term3 = np.exp(np.clip(exponent, -np.inf, 700))

    return term1 * term2 * term3

def grad_h_func(epsilon, sigma, h_func, k=K_PARAM):
    """
        h_func = the h function and we are finding the gradient as a multiple of it
        epsilon = standard normal distributed variable
        sigma = the std for g, and we are trying various values to find the optimal
        one that minimizes the variance after importance sampling
        k = a variable that determines the moment: n = 2k, <X^n> = <X^{2k}> is an even moment
    """

    term1 = (4*k + 2) / sigma
    term2 = 2 * sigma * np.power(epsilon, 2)
    gradient = (term1 - term2) * h_func
    return gradient

def get_optimal_sigma(k):
    """
        Returns the analytical optimal sigma for a given k.
    """
    return np.sqrt(2*k + 1)

#-------  Main program -----------------------

N     = 50000 # number of runtime

# Goal 1: Checking the effect of learning rate on getting optimal sigma
# for X^2

sigma = 1.5
learning_rate = [0.0001, 0.0005, 0.001]

rng   = np.random.default_rng(17)      # instantiate a random number generator object
epsilons = rng.standard_normal(N)      # instantiate all the samples at once

sigma_history = []
loss_history = []

for lr in learning_rate:
    tl = f"\nlearning_rate: {lr}"
    print(tl)
    print('-'*100)
    sigma_current = sigma

    for i in range(N):

        eps = epsilons[i]

        h_val = h_func(sigma_current, eps)
        grad_val = grad_h_func(sigma_current, eps, h_val)

        sigma_current = sigma_current - lr * grad_val       #formula from Asmussen and Glynn

        if sigma_current < 0.75:
            sigma_current = 0.75
        sigma_history.append(sigma_current)
        loss_history.append(h_val)

        if i % 1000 == 0:
            if i > 100:
                avg_loss = np.mean(loss_history[-100:])
                print(f"Iteration {i}: sigma = {sigma_current:.4f}, Est.Loss (avg) = {avg_loss:.4f}")
            else:
                print(f"Iteration {i}: sigma = {sigma_current:.4f}")


# Goal 2:
k_max = 10       # n = 2k, <X^n> = <X^{2k}> is an even moment
results = {}

for k in range(1, k_max + 1):
    sigma_opt = get_optimal_sigma(k)
    h_vals = h_func(sigma_opt, epsilons, k)
    grad_vals = grad_h_func(sigma_opt, epsilons, h_vals, k)


    noise_variance = np.var(grad_vals)
    results[k] = noise_variance

    print(f"X^{2*k} (sigma_opt = {sigma_opt:.3f}): Gradient Variance = {noise_variance:.4e}")
