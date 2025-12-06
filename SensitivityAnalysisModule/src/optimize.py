#Code author: Yunru Zheng
#Date: December 3, 2025

import numpy as np
import math
from .model import get_optimal_sigma

N_SAMPLES = 50000
CLIP_THRESHOLD = 10000.0
RNG = np.random.default_rng(17)

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