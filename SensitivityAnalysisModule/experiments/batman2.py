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

from src.model import f_jax, softmax, phi, make_g_function, make_likelihood_ratio, \
 make_score_function, loss_function, gradient_loss, Robbins_Monro
from src.utils import neural_net_plot, batman_plot

N_SAMPLES = 10000000
RNG = np.random.default_rng(17)
CLIP_THRESHOLD = 100.0



#----------------------------Main Program---------------------------
# Goal1: Finding the distribution after neural net and check that the distribution
# matches the theoretical pdf

a = 1.0
b = 0.45
c = 2.0
d = 0.3

#a, b, c, d = 2.0, 0.45, 2.0, 0.15
#a, b, c, d = 1.5955062, 0.45257553, 2.5, 0.20656002
a, b, c, d = 1.5, 0.4, 1.0, 0.2

theta = jnp.array([a, b, c, d])
Z_space = np.linspace(-6.0, 6.0, 100)


# X as a function of Z (X vs Z)
neural_net_plot(Z_space, theta, phi)

# Plot histogram and pdf
batman_plot(Z_space, theta, f_jax, phi, N_SAMPLES, RNG)

# Goal2: Trying to find the optimized theta to get a batman graph

V = lambda z: z**4

learning_rate = 0.0001
batch_size = 1000
num_iterations = N_SAMPLES // batch_size
theta_history = []
loss_history = []

fast_score_fn = make_score_function(phi)
fast_L_fn = make_likelihood_ratio(f_jax, phi)

for i in range(num_iterations):
    # 1. Generate Batch
    z_sample = RNG.standard_normal(batch_size)

    # 2. Tracking the loss
    loss_vals = loss_function(z_sample, theta, f_jax, phi, V, fast_L_fn)
    mean_loss = jnp.mean(loss_vals)
    loss_history.append(mean_loss)

    # 3. Run Optimizer
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
        L_current = fast_L_fn(z_sample, theta)
        integral_estimate = jnp.mean(V(x_current)* L_current)
        print(f"Iter {i:>4}: Loss={mean_loss:.4f} | Estimate={integral_estimate:.4f}")

# --- PLOTTING RESULTS ---
theta_str = [f"{x:.2e}" for x in np.array(theta)]
print(f"\nFinal Theta:, {theta_str}")

# Plot Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.title("Variance Reduction Training")
plt.xlabel("Iteration")
plt.ylabel("Estimator Variance (Loss)")
plt.yscale('log') # Log scale helps see progress better
plt.grid(True)
plt.show()

# Goal3: Plotting the graph after optimization to see whether it's improved

# X as a function of Z (X vs Z)
neural_net_plot(Z_space, theta, phi)

# Plot histogram and pdf
batman_plot(Z_space, theta, f_jax, phi, N_SAMPLES, RNG)