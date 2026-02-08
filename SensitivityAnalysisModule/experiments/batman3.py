#Code author: Yunru Zheng
#Date: January 9, 2025

import numpy as np
import matplotlib.pyplot as plt
import math

import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
from jax.scipy.stats import norm
from scipy import stats

from src.model import (
    f_jax, phi, 
    make_g_function,
    make_likelihood_ratio, 
    make_score_function, 
    make_mixing_likelihood_ratio,
    make_mixing_score_function,
    loss_function, 
    Robbins_Monro
)
from src.utils import neural_net_plot, batman_plot, plot_diagnostic

N_SAMPLES = 10000000
RNG = np.random.default_rng(17)
CLIP_THRESHOLD = 100.0

#----------------------------Main Program---------------------------
# Goal 1: Finding the distribution after neural net and check that the distribution
# matches the theoretical pdf

a = 1.0
b = 0.45
c = 2.0
d = 0.3
#a, b, c, d = 2.0, 0.45, 2.0, 0.15
#a, b, c, d = 1.5955062, 0.45257553, 2.5, 0.20656002
#a, b, c, d = 1.5, 0.4, 1.0, 0.2

a, b, c, d = 2.0, 0.5, 1.0, 0.2
#

theta = jnp.array([a, b, c, d])
Z_space = np.linspace(-6.0, 6.0, 100)


# X as a function of Z (X vs Z)
#neural_net_plot(Z_space, theta, phi)

# Plot histogram and pdf
#batman_plot(Z_space, theta, f_jax, phi, N_SAMPLES, RNG)

# Goal 2: Trying to find the optimized theta to get a batman graph

V = lambda z: z**4

learning_rate = 0.0005
mixing_ratio = 0.5

batch_size = 1000
num_iterations = N_SAMPLES // batch_size
theta_history = []
loss_history = []
smoothed_loss_history = []

# For training
fast_score_fn = make_mixing_score_function(phi, 0.1)
fast_L_fn = make_mixing_likelihood_ratio(f_jax, phi, 0.1)

# For Estimation of the mean
true_L_fn = make_likelihood_ratio(f_jax, phi)

# Initializing the tracking of the best parameter
best_smoothed_loss = float('inf')
best_theta = theta
best_iter = 0
smoothed_loss = None
smoothing_factor = 0.05

for i in range(num_iterations):
    # 1. Generate Batch
    z_sample = RNG.standard_normal(batch_size)

    # 2. Tracking the loss
    loss_vals = loss_function(z_sample, theta, f_jax, phi, V, fast_L_fn)
    mean_loss = jnp.mean(loss_vals)
    loss_history.append(mean_loss)

    # 3. Calculate Smoothed Loss
    if smoothed_loss is None:
        smoothed_loss = mean_loss
    else:
        # Exponential Moving Average (EMA)
        smoothed_loss = (1 - smoothing_factor) * smoothed_loss + smoothing_factor * mean_loss

    smoothed_loss_history.append(smoothed_loss)

    # 4. Check for best theta
    if smoothed_loss < best_smoothed_loss:
        best_smoothed_loss = smoothed_loss
        best_theta = theta
        best_iter = i

    # 5. Run Optimizer
    theta_new = Robbins_Monro(
        z_sample,
        theta,
        learning_rate,
        f_jax,
        phi,
        V,
        fast_L_fn,
        fast_score_fn,
        CLIP_THRESHOLD
    )

    theta = theta_new
    theta_history.append(theta)

    if i % 1000 == 0:
        print(f"Iter {i:>4}: Loss={mean_loss:.4f} | Theta={theta}")
        # Calculate the actual estimate of the integral: Mean( V(x) * L(x) )
        # This should be close to 3.0
        x_current = phi(z_sample, theta)
        L_current = true_L_fn(z_sample, theta)
        integral_estimate = jnp.mean(V(x_current)* L_current)
        print(f"Iter {i:>4}: Loss={mean_loss:.4f} | Estimate={integral_estimate:.4f}")

print(f"\nFinal Theta:, {theta}, learning rate: {learning_rate}")
print(f"\nOptimal Theta, {best_theta}")

# ---------- PLOTTING RESULTS ------------

# 1. Process Data
block_size = 100

binned_loss = [np.mean(loss_history[i:i+block_size])
            for i in range(0, len(loss_history), block_size)]

# 2. Plot the Raw Data
plt.figure(figsize=(10, 5))
plt.plot(loss_history, color='lightblue', alpha=0.5, label='Raw Batch Noise')


'''
previously used smoothed data
# 1. Process Data (Your original plotting logic)
block_size = 100
binned_loss = [np.mean(loss_history[i:i+block_size])
            for i in range(0, len(loss_history), block_size)]
'''

# 3. Plot the Smoothed Trend
plt.plot(smoothed_loss_history, color='darkblue', linewidth=1.5, label='Smoothed Trend')
plt.scatter(best_iter, best_smoothed_loss, color='red', s = 150, marker='*', label=f'Best Early Stop (Iter {best_iter})')

plt.yscale('log')
plt.title("Variance Reduction: Noise vs Trend")
plt.legend()
plt.show()


# Goal3: Plotting the graph after optimization to see whether it's improved

# Plot the best theta first
batman_plot(Z_space, best_theta, f_jax, phi, N_SAMPLES, RNG)


# X as a function of Z (X vs Z)
neural_net_plot(Z_space, theta, phi)

# Plot histogram and pdf
batman_plot(Z_space, theta, f_jax, phi, N_SAMPLES, RNG)


# Goal4: Checking the zero variance estimator

# plot_diagnostic(theta, phi, f_jax, z_range=(-6, 6), n_points=1000)