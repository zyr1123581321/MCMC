#Code author: Yunru Zheng
#Date: December 3, 2025

import numpy as np
import math
import matplotlib.pyplot as plt
from .model import get_optimal_sigma, make_g_function

# --- JAX (Automatic Differentiation & Compilation) ---
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
from jax.scipy.stats import norm


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


# Batman importance sampling

def neural_net_plot(Z_space, theta, phi):
    """
    Plotting the results going through a neural net (X vs Z)
    Input:
        Z_space: The linspace of Z
        theta: Parameters for the transformation function
        phi: Transformation function phi(z, theta)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(Z_space, phi(Z_space, theta), label='estimation variance')
    plt.title(f"The effect of the neural net on Z")
    plt.xlabel("Z")
    plt.ylabel("X")
    plt.grid(True, alpha=0.3)

    plt.legend()
    plt.show()

def batman_plot(Z_space, theta, f, phi, n_samples, rng):
    """
    Plotting the batman plot given the parameters
    Input:
        Z_space: The linspace of Z
        theta: Parameters for the transformation function
        f: pdf of Z
        phi: Transformation function phi(z, theta)
        n_samples: Total sample size
        rng: the random number seed
    """
    Z = rng.standard_normal(n_samples)
    X = phi(Z, theta)
    fast_g_fn = make_g_function(f, phi)

    X_theoretical = phi(Z_space, theta)
    Y_theoretical = fast_g_fn(Z_space, theta)

    plt.figure(figsize=(10, 6))

    # The histogram of X
    plt.hist(X, bins=100, density=True, alpha=0.5, label="Empirical Histogram")

    # The pdf of X
    plt.plot(X_theoretical, Y_theoretical, label="Theoretical pdf")
    plt.title(f"The distribution of the neural net (N={n_samples})")
    plt.xlabel("X")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)

    plt.legend()
    plt.show()

def plot_diagnostic(theta, phi, f, z_range=(-6, 6), n_points=1000):
    """
    Plots the "Zero Variance" diagnostic: y = x^4 * (f(x) / g(x))
    A perfect model would result in a flat line at y = 3.0.
    """
    # 1. Prepare Data
    z_space = jnp.linspace(z_range[0], z_range[1], n_points)

    # 2. Define the components locally to ensure they match your logic
    phi_grad = grad(phi, argnums=0) # dphi/dz

    def get_diagnostic_value(z):
        # Transform z to x
        x = phi(z, theta)

        # Compute g(x) = f(z) / |phi'(z)|
        slope = jnp.abs(phi_grad(z, theta))
        g_val = f(z) / slope

        # Compute f(x) (Target density at x)
        f_val = f(x)

        # Likelihood Ratio: L = f(x) / g(x)
        likelihood_ratio = f_val / g_val

        # The Observable: h(x) = x^4
        observable = x**4

        # The Product: x^4 * L
        return x, observable * likelihood_ratio

    # Vectorize the calculation
    vmap_diag = vmap(get_diagnostic_value)
    x_vals, y_vals = vmap_diag(z_space)

    # 3. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, color='purple', linewidth=2, label=r'Diagnostic: $x^4 \frac{f(x)}{g(x)}$')

    # Add the "Perfect" reference line at y=3
    plt.axhline(y=3.0, color='black', linestyle='--', alpha=0.5, label='Optimal (Constant = 3)')

    # Formatting
    plt.title("Zero Variance Diagnostic Check")
    plt.xlabel("x (Transformed Value)")
    plt.ylabel("Weighted Value")
    plt.yscale('log') # Log scale is crucial because spikes can be huge
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.show()

# --- How to run it ---
# plot_diagnostic(theta, phi, f_jax)