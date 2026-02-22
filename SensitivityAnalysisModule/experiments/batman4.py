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
    gradient_loss,
    Robbins_Monro
)
from src.utils import (
    neural_net_plot,
    batman_plot,
    plot_smooth_loss,
    plot_diagnostic,
    plot_gradient_snapshot
)
N_SAMPLES = 20000000
RNG = np.random.default_rng(17)
CLIP_THRESHOLD = None

LEARNING_RATE = 0.0005
MIXING_RATIO = 0.1
TRIMMING_RATIO = 0.0

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


batch_size = 10000
num_iterations = N_SAMPLES // batch_size
theta_history = []
loss_history = []
smoothed_loss_history = []

# Initializing the gradient loss trackers
grad_history = {
    'start': None,
    'best': None,
    'final': None
}

grads_best = None

# For training
fast_score_fn = make_mixing_score_function(phi, MIXING_RATIO)
fast_L_fn = make_mixing_likelihood_ratio(f_jax, phi, MIXING_RATIO)

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

    # 5. Run Optimizer
    theta_new, raw_mean_grad, avg_grad, current_raw_grads = Robbins_Monro(
        z_sample,
        theta,
        LEARNING_RATE,
        f_jax,
        phi,
        V,
        fast_L_fn,
        fast_score_fn,
        CLIP_THRESHOLD,
        TRIMMING_RATIO
    )

    if i == 0:
        grad_history['start'] = {'iter': i, 'grads': current_raw_grads}

    if i == num_iterations - 1:
        grad_history['final'] = {'iter': i, 'grads': current_raw_grads}

    # 4. Check for best theta
    if smoothed_loss < best_smoothed_loss:
        best_smoothed_loss = smoothed_loss
        best_theta = theta
        best_iter = i
        grads_best = current_raw_grads

    theta = theta_new
    theta_history.append(theta)

    if (i < 500 and i % 30 == 0) or (i >= 300 and i % 1000 == 0):
        # Calculate the actual estimate of the integral: Mean( V(x) * L(x) )
        # This should be close to 3.0
        x_current = phi(z_sample, theta)
        L_current = true_L_fn(z_sample, theta)
        integral_estimate = jnp.mean(V(x_current)* L_current)

        # Print theta AND the gradient magnitude
        grad_mag = np.linalg.norm(raw_mean_grad)
        print(f"Iter {i:>4}: Loss={mean_loss:.4f} | Est={integral_estimate:.4f}")
        print(f"           Theta: {theta}")
        '''
        print(f"           Raw Grad : {raw_mean_grad} (Mag: {grad_mag:.2f})")
        print(f"           Trimmed Grad : {avg_grad}")
        '''
print(f"\nFinal Theta:, {theta}, learning rate: {LEARNING_RATE}, mixing ratio: {MIXING_RATIO}")
print(f"\nOptimal Theta, {best_theta}")

grad_history['best'] = {'iter': best_iter, 'grads': grads_best}



# ---------- PLOTTING RESULTS ------------

# Plot the histogram of the raw gradient loss
param_names = ['a', 'b', 'c', 'd']

for stage_name, data in grad_history.items():
    # Unpack data from the dictionary
    iteration = data['iter']
    gradients = data['grads']

    # Create labels for each stage
    label = f"{stage_name} (Iter = {iteration})"

    plot_gradient_snapshot(gradients, label, param_names, batch_size, TRIMMING_RATIO, CLIP_THRESHOLD)


# 1. Plot the Raw Data
plt.figure(figsize=(10, 5))
plt.plot(loss_history, color='lightblue', alpha=0.5, label='Raw Batch Noise')


'''
previously used smoothed data
# 1. Process Data (Your original plotting logic)
block_size = 100
binned_loss = [np.mean(loss_history[i:i+block_size])
            for i in range(0, len(loss_history), block_size)]
'''

# 2. Plot the Smoothed Trend
plot_smooth_loss(smoothed_loss_history, best_iter, best_smoothed_loss, TRIMMING_RATIO, CLIP_THRESHOLD)


# Goal3: Plotting the graph after optimization to see whether it's improved

# Plot the best theta first
batman_plot(Z_space, best_theta, f_jax, phi, N_SAMPLES, RNG)


# X as a function of Z (X vs Z)
neural_net_plot(Z_space, theta, phi)

# Plot histogram and pdf
batman_plot(Z_space, theta, f_jax, phi, N_SAMPLES, RNG)


# Goal4: Checking the zero variance estimator

# plot_diagnostic(theta, phi, f_jax, z_range=(-6, 6), n_points=1000)