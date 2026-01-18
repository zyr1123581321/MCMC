#Code author: Yunru Zheng
#Date: December 3, 2025

import numpy as np
import math
from scipy import stats
from .model import get_optimal_sigma

N_SAMPLES = 50000
CLIP_THRESHOLD = 10000.0
RNG = np.random.default_rng(17)

# HELPER FUNCTION:
# is_reparam = True corresponds to Z method, and False for X method

def run_sgd_optimization(k, grad_func, loss_func, batch_size,
    learning_rate, is_reparam=True, num_iterations=None, start_sigma=None, n_samples=50000):
    if start_sigma is None:
        sigma_current = np.sqrt(2 * k) # Default heuristic start
    else:
        sigma_current = start_sigma    # Manual override (e.g., 1.0 for Ex 4)

    if num_iterations is None:
        # Default to full epoch if not specified
        # Assumes N_SAMPLES is a global constant, or you can add it as an arg
        num_iterations = n_samples // batch_size

    sigma_opt = get_optimal_sigma(k)    # the optimal value sigma = \sqrt{2k+1}

    sigma_history = []
    loss_history = []

    # Calculate how many steps to make based on batch size
    # Batch=1   -> 50,000 steps
    # Batch=50  ->  1,000 steps

    for i in range(num_iterations):
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

        avg_grad = stats.trim_mean(valid_grads, 0.2)
        avg_grad = np.clip(avg_grad, -CLIP_THRESHOLD, CLIP_THRESHOLD)

        # 4. Update
        sigma_current = sigma_current - learning_rate * avg_grad

        if sigma_current < 0.75:
            sigma_current = 0.75

        sigma_history.append(sigma_current)
        loss_history.append(np.mean(loss_vals))
    return sigma_history, loss_history