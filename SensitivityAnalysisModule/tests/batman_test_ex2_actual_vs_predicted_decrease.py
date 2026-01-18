#Code author: Yunru Zheng
#Date: December 6, 2025

import numpy as np
from src.utils import var_estimator
from src.model import f_jax, phi, make_likelihood_ratio, \
 make_score_function, loss_function, gradient_loss

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

RNG = np.random.default_rng(17)
n_samples = 1000000

#---------------------------Main------------------------------
# Goal: finding the good step size so predicted decrease is \Omega(actual decrease)
# a.k.a. c_1 * actual_decrease \leq predicted_decrease \leq c_2 * actual_decrease
# Actual decrease: u(theta_{n+1}) - u(theta_n)
# Predicted decrease = grad u(theta_n) * dtheta = step_size * |\grad u(theta_n)|^2

# step size results in different x_{n+1} since
# theta_{n+1} = theta_n + step_size * \grad u(theta_n)
step_sizes = [0.005, 0.001, 0.0005, 1e-4, 1e-5, 1e-6, 1e-7]

# Setting up functions for later use
fast_score_fn = make_score_function(phi)
fast_L_fn = make_likelihood_ratio(f_jax, phi)

# Batch size = 1000000
z_batch = RNG.standard_normal(n_samples)

# The function we test on is X^4
V = lambda z: z**4
theta = jnp.array([2.0, 0.45, 2.0, 0.15])

# Old loss function
loss_vals = loss_function(z_batch, theta, f_jax, phi, V, fast_L_fn)
loss_old = jnp.mean(loss_vals)

# Gradient loss for the initial theta value
grad_loss = gradient_loss(z_batch, theta, f_jax, phi, V, fast_L_fn, fast_score_fn)

# As I choose a very large sample size for the gradient loss (batch size = 100000 here)
# The mean value approaches the actual gradient!
estimated_grad_var = jnp.mean(grad_loss, axis=0)

# Calculating norm squared: |grad u(theta_n)|^2
grad_norm_sq = jnp.sum(estimated_grad_var**2)

print(f"{'Step Size':<10} | {'Pred Decrease':<15} | {'Actual Decrease':<15} | {'Ratio (Act/Pred)'}")

for s in step_sizes:
    theta_new = theta - s * estimated_grad_var

    loss_new = jnp.mean(loss_function(z_batch, theta_new, f_jax, phi, V, fast_L_fn))

    # Actual decrease: u(theta_{n+1}) - u(theta_n)
    actual_decrease = jnp.abs(loss_new - loss_old)

    predicted_decrease = s * grad_norm_sq

    # Ratio Check
    ratio = actual_decrease / predicted_decrease

    print(f"{s:<10.2e} | {float(predicted_decrease):<15.4e} | {float(actual_decrease):<15.4e} | {float(ratio):<15.4f}")



