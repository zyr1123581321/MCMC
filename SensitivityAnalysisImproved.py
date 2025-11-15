#Code author: Yunru Zheng
#Date: November 14, 2025

import numpy as np
import matplotlib.pyplot as plt
import math


# We are doing sensitivity analysis on the std of our importance function g(x) = N(0,sigma).
# We will find the gradient of the variance with respect to sigma, and use statistical
# approximation to find the optimal sigma value by gradient descent.

K_PARAM = 1

def m_func(X, sigma, k=K_PARAM):
    """
        x = N(0, sigma)
        sigma = the std for g, and we are trying various values to find the optimal
        one that minimizes the variance after importance sampling
        k = a variable that determines the moment: n = 2k, <X^n> = <X^{2k}> is an even moment
    """
    exponent = -np.power(X,2) * (1-1/np.power(sigma,2))
    term1 = np.power(sigma,2) * np.exp(exponent)
    term2 = np.power(X, 4*k)
    return term1 * term2

def grad_m_func(x, sigma, m_func, k=K_PARAM):
    """
        x = N(0, sigma)
        sigma = the std for g, and we are trying various values to find the optimal
        one that minimizes the variance after importance sampling
        m_func = the m function and we are finding the gradient as it multiplying some value
        k = a variable that determines the moment: n = 2k, <X^n> = <X^{2k}> is an even moment
    """
    term1 = (np.power(sigma, 2) - np.power(x,2)) / np.power(sigma, 3)
    return term1 * m_func


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

def get_optimal_sigma(k):
    """
        Returns the analytical optimal sigma for a given k.
    """
    return np.sqrt(2*k + 1)

#-------  Main program -----------------------

N= 20000 # number of runtime

# Goal 1: Checking the effect of learning rate on getting optimal sigma
# for X^2

sigma         = 1.5
learning_rates = [0.0005, 0.001, 0.005]

rng      = np.random.default_rng(17)      # instantiate a random number generator object
Zs       = rng.standard_normal(N)      # instantiate all the samples at once

sigma_history_x = []    #storage for sigma using x
loss_history_x = []     #storage for the function using x

sigma_history_z = []    #storage for sigma using z
loss_history_z = []     #storage for the function using z

# Using Prof. Goodman's approach to differentiate the integrand directly and using
# X ~ N(0, sigma)

for lr in learning_rates:
    tl = f"\n Direct Gradient Descent using X ~ N(0, sigma) | batch size = 1 | learning_rate: {lr}"
    print(tl)
    print('-'*100)
    sigma_current = sigma

    for i in range(N):

        X = rng.standard_normal() * sigma_current

        m_val = m_func(X, sigma_current)
        grad_val = grad_m_func(X, sigma_current, m_val)

        sigma_current = sigma_current - lr * grad_val       #formula from Asmussen and Glynn

        if sigma_current < 0.75:
            sigma_current = 0.75    #explained in my filed sent to you

        # Recording the values for later use
        sigma_history_x.append(sigma_current)
        loss_history_x.append(m_val)

        # Trying to find a pattern by snapshotting the sigma value and the average of h_val
        # after 1000 runs
        if i % 1000 == 0:
            if i > 100:
                avg_loss = np.mean(loss_history_x[-100:])
                print(f"Iteration {i}: sigma = {sigma_current:.4f}, Est.Loss (avg) = {avg_loss:.4f}")
            else:
                print(f"Iteration {i}: sigma = {sigma_current:.4f}")

# Using my approach to set X = sigma*Z and differentiate the integrand
# using Z ~ N(0, 1)


for lr in learning_rates:
    tl = f"\n Direct Gradient Descent using Z ~ N(0, 1) | batch size = 1 | learning_rate: {lr}"
    print(tl)
    print('-'*100)
    sigma_current = sigma

    for i in range(N):

        Z = rng.standard_normal()

        h_val = h_func(Z, sigma_current)
        grad_val = grad_h_func(Z, sigma_current, h_val)

        sigma_current = sigma_current - lr * grad_val       #formula from Asmussen and Glynn

        if sigma_current < 0.75:
            sigma_current = 0.75    #explained in my filed sent to you

        # Recording the values for later use
        sigma_history_z.append(sigma_current)
        loss_history_z.append(h_val)

        # Trying to find a pattern by snapshotting the sigma value and the average of h_val
        # after 1000 runs
        if i % 1000 == 0:
            if i > 100:
                avg_loss = np.mean(loss_history_z[-100:])
                print(f"Iteration {i}: sigma = {sigma_current:.4f}, Est.Loss (avg) = {avg_loss:.4f}")
            else:
                print(f"Iteration {i}: sigma = {sigma_current:.4f}")


# Goal 2: I'm going to use Batch Gradient Descent to see if the results would be more stable. After looking at the results from step 1, I decided to use step side 0.0005.

sigma_current = 1.7
N_batch = 50000

learning_rate = 0.005
batch_size = 50
num_iternations = N_batch // batch_size

sigma_history_batchx = []    #storage for sigma using batch x
loss_history_batchx = []     #storage for the function using batch x

sigma_history_batchz = []    #storage for sigma using batch z
loss_history_batchz = []     #storage for the function using batch z

tl1 = f"\nBatch Gradient Descent using X ~ N(0, sigma) | batch size = {batch_size} |"
tl1 = tl1 + f" learning_rate: {learning_rate}"
print(tl1)
print('-'*100)


for i in range(num_iternations):
    X_batch = rng.standard_normal(batch_size) * sigma_current

    m_vals_batch = m_func(X_batch, sigma_current)
    grad_vals_batch = grad_m_func(X_batch, sigma_current, m_vals_batch)

    valid_m_vals = m_vals_batch[~(np.isinf(m_vals_batch) | np.isnan(m_vals_batch))]
    valid_grads = grad_vals_batch[~(np.isinf(grad_vals_batch) | np.isnan(grad_vals_batch))]

    if not valid_grads.any(): # Check if the batch was empty after cleaning
        print(f"Iteration {i}: All samples in batch overflowed. Skipping step.")
        continue

    avg_gradient = np.mean(valid_grads)
    sigma_current = sigma_current - learning_rate * avg_gradient

    if sigma_current < 0.75:
        sigma_current = 0.75

    sigma_history_batchx.append(sigma_current)
    loss_history_batchx.append(np.mean(m_vals_batch))

    if i % 50 == 0:
        tl = f"Iteration {i}: sigma = {sigma_current:.4f}, "
        tl = tl + f"Est.Loss(avg) = {loss_history_batchx[-1]:.4f}"
        print(tl)
print(f"Final sigma (avg of last 50 steps): {np.mean(sigma_history_batchx[-50:]):.4f}")


tl1 = f"\nBatch Gradient Descent using Z ~ N(0, 1) | batch size = {batch_size} |"
tl1 = tl1 + f" learning_rate: {learning_rate}"
print(tl1)
print('-'*100)


for i in range(num_iternations):
    Z_batch = rng.standard_normal(batch_size)

    h_vals_batch = h_func(Z_batch, sigma_current)
    grad_vals_batch = grad_h_func(Z_batch, sigma_current, h_vals_batch)

    valid_h_vals = h_vals_batch[~(np.isinf(h_vals_batch) | np.isnan(h_vals_batch))]
    valid_grads = grad_vals_batch[~(np.isinf(grad_vals_batch) | np.isnan(grad_vals_batch))]

    if not valid_grads.any(): # Check if the batch was empty after cleaning
        print(f"Iteration {i}: All samples in batch overflowed. Skipping step.")
        continue

    avg_gradient = np.mean(valid_grads)
    sigma_current = sigma_current - learning_rate * avg_gradient

    if sigma_current < 0.75:
        sigma_current = 0.75

    sigma_history_batchz.append(sigma_current)
    loss_history_batchz.append(np.mean(h_vals_batch))

    if i % 50 == 0:
        tl = f"Iteration {i}: sigma = {sigma_current:.4f}, "
        tl = tl + f"Est.Loss(avg) = {loss_history_batchz[-1]:.4f}"
        print(tl)
print(f"Final sigma (avg of last 50 steps): {np.mean(sigma_history_batchz[-50:]):.4f}")



# Goal 3: I'm trying to show that I cannot find reliable optimal sigma value as the moment gets larger
# by showing that the gradient noise gets big and very deviated from the true value

k_max = 10       # n = 2k, <X^n> = <X^{2k}> is an even moment
results = []

for k in range(1, k_max + 1):
    N = 50000
    learning_rate = 0.001
    batch_size = 50
    num_iternations = N // batch_size
    sigma_history = []

    sigma_opt = get_optimal_sigma(k)
    sigma_current = np.sqrt(2 * k)
    for i in range(num_iternations):

        Z_batch = rng.standard_normal(batch_size)

        h_vals_batch = h_func(Z_batch, sigma_current, k)
        grad_vals_batch = grad_h_func(Z_batch, sigma_current, h_vals_batch, k)

        valid_grads = grad_vals_batch[~(np.isinf(grad_vals_batch) | np.isnan(grad_vals_batch))]

        if not valid_grads.any():
            print(f"k={k}, i={i}: All grads in batch overflowed! Skipping.")
            continue

        avg_gradient = np.mean(valid_grads)
        clip_threshold = 10000
        avg_gradient = np.clip(avg_gradient, -clip_threshold, clip_threshold)

        sigma_current = sigma_current - learning_rate * avg_gradient

        if sigma_current < 0.75:
            sigma_current = 0.75

        sigma_history.append(sigma_current)

    sigma_final = np.mean(sigma_history[-50:])
    abs_error = abs(sigma_final - sigma_opt)
    results.append((k, sigma_final, sigma_opt, abs_error))

print("\n" + "="*60)
print("--- FINAL RESULTS SUMMARY ---")
print("="*60)
print(f"{'n (X^2n)':<10} | {'Sigma':<15} | {'Optimal Sigma':<15} | {'Absolute Error':<15}")
print("-"*60)

for res in results:
    n, found, opt, err = res
    print(f"{n:<10} | {found:<15.4f} | {opt:<15.4f} | {err:<15.4f}")