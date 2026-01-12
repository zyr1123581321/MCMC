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

from src.model import f
from src.utils import print_optimization_process, print_results_table

N_SAMPLES = 10000000
RNG = np.random.default_rng(17)
CLIP_THRESHOLD = 100.0

def h(t):
    """
    The softmax sigma function. I used h because symbol sigma was used
    Input:
        t: independent variable
    Output:
        h(t) = log(1 + e^t)
    """
    return jnp.log(1 + jnp.exp(t))

def phi(z, theta):
    """
    The transition function that outputs new variable X from Z
    Input:
        z: the input random variable
        a, b, c, d = theta
        a: the initial slope before the turn
        b: controls the slope after the turn together with d
        c: the turning point
        d: the width of the turn
    Output:
        phi(z, theta) = 2*a*z - b(h((z-c)/d) - h((-z-c)/d))
    """
    a, b, c, d = theta
    return 2*a*z - b*(h((z-c)/d) - h((-z-c)/d))

def f(z):
    """
    pdf of Z
    Input:
        z: original random variable
    Output:
        standard gaussian distribution
    """
    return norm.pdf(z)

def make_g_function(f, phi):
    """
    g is the pdf of the new variable X
    Input:
        f: pdf of random variable Z
        phi: the transformation function
    Output:
        Vectorized g
    """
    # Define partial derivative of phi wrt z
    phi_z = grad(phi, argnums=0)

    def compute_g_single(z, theta):
        """
        Compute g for each individual z value
        Input:
            z: random variable samples
            theta: Parameters for the transformation function
        Output:
            g(x)= f(z)/|phi_z|
        """
        # Calculate the slope |phi_z|
        slope = jnp.abs(phi_z(z, theta))
        denom = f(z)
        return denom / slope

    # Vectorize g function
    g_batch = vmap(compute_g_single, in_axes=(0, None))
    return jit(g_batch)


def make_likelihood_ratio(f, phi):
    """
    The ratio between the distribution before and after neural net
    Input:
        f: pdf of random variable Z
        phi: the transformation function
    Output:
        Vectorized L
    """
    phi_z = grad(phi, argnums=0)

    def compute_L_single(z, theta):
        """
        Compute L for each individual z value
        Input:
            z: random variable samples
            theta: Parameters for the transformation function
        Output:
            L(x, theta) = f(x) / g(x, theta) = f(x) * |phi_z| / f(z)
        """
        # Calculate x and the slope
        x = phi(z, theta)
        # slope = |phi_z|
        slope = jnp.abs(phi_z(z, theta))

        # log(L) = log(f(x)) + log(|phi_z|) - log(f(z))
        log_fx = -0.5 * (x**2)
        log_fz = -0.5 * (z**2)
        log_slope = jnp.log(slope + 1e-10)

        log_L = log_fx + log_slope - log_fz

        return jnp.exp(log_L)

    L_batch = vmap(compute_L_single, in_axes=(0, None))
    return jit(L_batch)

def make_score_function(phi):
    """
    The score function is the additional function generate after taking
    derivative of the variance
    Input:
        phi: the transformation function phi(z, theta)
    Output:
        Vectorized score function
    """
    # --- 1. DEFINE DERIVATIVES (Only once)---
    # First Derivatives
    # phi_z: (d_phi / d_z)
    phi_z_fn = grad(phi, argnums=0)

    # phi_theta: vector of 4 (d_phi / d_theta)
    phi_theta_fn = grad(phi, argnums=1)

    # Second Derivatives
    # phi_zz: (d^2_phi / d_z^2)
    phi_zz_fn = grad(phi_z_fn, argnums=0)

    # phi_z_theta: vector of 4 (d_phi_z / d_theta)
    phi_z_theta_fn = grad(phi_z_fn, argnums=1)

    def compute_score_single(z, theta):
        """
        Input:
            z: random variable samples
            theta: Parameters for the transformation function
        Output:
            S_total = -(S_I + S_II))
            = -z * (phi_theta/phi_z) - phi_zz*phi_theta/(phi_z)^2 - phi_z_theta/phi_z
        """
        phi_z = phi_z_fn(z, theta)
        phi_theta = phi_theta_fn(z, theta)
        phi_zz = phi_zz_fn(z, theta)
        phi_z_theta = phi_z_theta_fn(z, theta)

        # Compute Score function
        # S_I = -(f_z/f) * (phi_theta/phi_z)
        # In fact -(f_z/f) = z
        term1 = z
        S_I = term1 * (phi_theta/phi_z)

        # S_II = phi_zz*phi_theta/(phi_z)^2 - phi_z_theta/phi_z
        term_a = phi_zz*phi_theta / (phi_z)**2
        term_b = phi_z_theta/phi_z
        S_II = term_a - term_b

        # S_total = -(S_I + S_II)
        S_total = -(S_I + S_II)

        return S_total

    # compute_derivatives_batch vectorizes that
    score_batch = vmap(compute_score_single, in_axes=(0, None))

    return jit(score_batch)

def loss_function(z_batch, theta, f, phi, V, L):
    """
    The loss (V(x)^2 * L(x, theta)^2) of the variable X
    Input:
        z_batch: Original random variable samples (single value if batch size = 1)
        theta: Parameters for the transformation function
        f: pdf of Z
        phi: Transformation function phi(z, theta)
        V: Function V(x)
        L: Likelihood ratio function L(x, theta)
    Output:
        loss = V(x)^2*L(x, theta)^2
    """
    # 1. Transform Z -> X
    x_batch = phi(z_batch, theta)

    # 2. Compute V(x)
    V_val = V(x_batch)

    # 3. Compute Likelihood Ratio L
    L_batch = L(z_batch, theta)

    # 4. Compute the Loss Function u
    loss = (V_val**2) * (L_batch**2)

    return loss

def gradient_loss(z_batch, theta, f, phi, V, L, score_fn):
    """
    The gradient loss (G(x, theta) = V(x)^2 * L(x, theta)^2 * S(x, theta)) of the variable X
    Input:
        z_batch: Original random variable samples (single value if batch size = 1)
        theta: Parameters for the transformation function
        f: pdf of Z
        phi: Transformation function phi(z, theta)
        V: Function V(x)
        L: Likelihood ratio function L(x, theta)
        score_fn: Score function S(x, theta)
    Output:
        raw_gradients = V(x)^2*L(x, theta)^2
    """
    # 1. Transform Z -> X
    x_batch = phi(z_batch, theta)

    # 2. Compute V(x) and convert it from (N,) to (N, 1)
    V_val = V(x_batch)[:, None]

    # 3. Compute Likelihood Ratio L and convert it from (N,) to (N, 1)
    L_batch = L(z_batch, theta)[:, None]

    # 4. Compute Score Function S: (N, 4)
    S_batch = score_fn(z_batch, theta)

    # 5. G(x, theta) = V(x)^2*L(x, theta)^2 * S(x, theta)
    raw_gradients = (V_val**2) * (L_batch**2) * S_batch

    return raw_gradients

def Robbins_Monro(z_batch, theta, step_size, f, phi, V, L, score_fn):
    """
    Gradient descent algorithm that uses the the gradient loss instead of the expected value
    theta_{n+1} = theta_n - step_size * G(x, theta_n)
    Constraints are given to restrict the new theta value
    Input:
        z_batch: Original random variable samples (single value if batch size = 1)
        theta: Parameters for the transformation function
        step_size: Learning rate
        f: pdf of Z
        phi: Transformation function phi(z, theta)
        V: Function V(x)
        L: Likelihood ratio function L(x, theta)
        score_fn: Score function S(x, theta)
    Output:
        updated theta
    """
    # 1. Get raw gradient data
    raw_grads = np.array(gradient_loss(z_batch, theta, f, phi, V, L, score_fn))

    # 2. Data Cleaning
    valid_mask = np.all(np.isfinite(raw_grads), axis=1)
    clean_grads = raw_grads[valid_mask]

    # If nothing left, skip the step
    if len(clean_grads) == 0:
        print("Warning: All gradients exploded. Skipping step.")
        return theta

    # 3. Trimmed Mean (For now getting rid of top 20% and bottom 20%)
    avg_grad = stats.trim_mean(clean_grads, 0.2)

    # 4. Clipping
    avg_grad = np.clip(avg_grad, -CLIP_THRESHOLD, CLIP_THRESHOLD)

    # 5. Update theta
    theta_new = theta - step_size * avg_grad

    # 6. CONSTRAINT CHOICES
    a_new, b_new, c_new, d_new = theta_new

    # Underconstruction because the choices of constraints here affect
    # how optimal theta is

    # Constraint A: c is large enough so X doesn't just cluster near 0
    c_new = jnp.maximum(c_new, 2.5)

    # Constraint B: a is large enough so the center is undersampled
    a_new = jnp.maximum(a_new, 1.0)

    # Constraint A: d > 0 (width > 0)
    # Also cannot be too small so it pushes a to be super large
    d_new = jnp.maximum(d_new, 0.01)

    # Attempt 1
    # max_b < d * (2 * a - 0.05)
    # max_b = d_new * (2 * a_new - 0.05)
    # b_new = jnp.minimum(b_new, max_b)

    # Constraint B:
    # 2a - b / d > 0.05 (Positive slope to ensure that
    # the neural net is one-to-one)

    # Attempt 2
    # 2a - b / d >= 1
    min_a = 0.5 + b_new / (2.0 * d_new)
    a_new = jnp.maximum(a_new, min_a)

    # (Optional) Constraint C: b > 0 (slope decreases rather than increases)
    b_new = jnp.maximum(b_new, 0.1)

    # 7. Formulating final theta
    theta_final = jnp.array([a_new, b_new, c_new, d_new])

    return theta_final

def neural_net_plot(Z_space, theta):
    """
    Plotting the results going through a neural net (X vs Z)
    Input:
        Z_space: The linspace of Z
        theta: Parameters for the transformation function
    """
    plt.figure(figsize=(10, 6))
    plt.plot(Z_space, phi(Z_space, theta), label='estimation variance')
    plt.title(f"The effect of the neural net on Z")
    plt.xlabel("Z")
    plt.ylabel("X")
    plt.grid(True, alpha=0.3)

    plt.legend()
    plt.show()

def batman_plot(Z_space, theta, f, phi, n_samples):
    Z = RNG.standard_normal(n_samples)
    X = phi(Z, theta)
    fast_g_fn = make_g_function(f, phi)

    X_theoretical = phi(Z_space, theta)
    Y_theoretical = fast_g_fn(Z_space, theta)

    plt.figure(figsize=(10, 6))

    # The histogram of X
    plt.hist(X, bins=100, density=True, alpha=0.5, label="Empirical Histogram")

    # The pdf of X
    plt.plot(X_theoretical, Y_theoretical, label="Theoretical pdf")
    plt.title(f"The distribution of the neural net (N={N_SAMPLES})")
    plt.xlabel("X")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)

    plt.legend()
    plt.show()


#----------------------------Main Program---------------------------
# Goal1: Finding the distribution after neural net and check that the distribution
# matches the theoretical pdf

a = 1.0
b = 0.45
c = 2.0
d = 0.3

a, b, c, d = 2.0, 0.45, 2.0, 0.15
#a, b, c, d = 1.5955062, 0.45257553, 2.5, 0.20656002

theta = jnp.array([a, b, c, d])
Z_space = np.linspace(-6.0, 6.0, 100)


# X as a function of Z (X vs Z)
neural_net_plot(Z_space, theta)

# Plot histogram and pdf
batman_plot(Z_space, theta, f, phi, N_SAMPLES)

# Goal2: Trying to find the optimized theta to get a batman graph

V = lambda z: z**4
f = lambda z: norm.pdf(z)

learning_rate = 0.005
batch_size = 1000
num_iterations = N_SAMPLES // batch_size
theta_history = []
loss_history = []

fast_score_fn = make_score_function(phi)
fast_L_fn = make_likelihood_ratio(f, phi)

for i in range(num_iterations):
    # 1. Generate Batch
    z_sample = RNG.standard_normal(batch_size)

    # 2. Tracking the loss
    loss_vals = loss_function(z_sample, theta, f, phi, V, fast_L_fn)
    mean_loss = jnp.mean(loss_vals)
    loss_history.append(mean_loss)

    # 3. Run Optimizer
    theta_new = Robbins_Monro(
        z_sample,
        theta,
        learning_rate,
        f,
        phi,
        V,
        fast_L_fn,
        fast_score_fn
    )

    theta = theta_new
    theta_history.append(theta)

    if i % 1000 == 0:
        print(f"Iter {i}: Loss={mean_loss:.4f} | Theta={theta}")
        # Calculate the actual estimate of the integral: Mean( V(x) * L(x) )
        # This should be close to 3.0
        x_current = phi(z_sample, theta)
        L_current = fast_L_fn(z_sample, theta)
        integral_estimate = jnp.mean(V(x_current)* L_current)
        print(f"Iter {i}: Loss={mean_loss:.4f} | Estimate={integral_estimate:.4f}")

# --- PLOTTING RESULTS ---
print("\nFinal Theta:", theta)

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
neural_net_plot(Z_space, theta)

# Plot histogram and pdf
batman_plot(Z_space, theta, f, phi, N_SAMPLES)