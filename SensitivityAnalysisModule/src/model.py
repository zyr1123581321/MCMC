#Code author: Yunru Zheng
#Date: December 3, 2025

import numpy as np
import math
from scipy import stats

# --- JAX (Automatic Differentiation & Compilation) ---
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
from jax.scipy.stats import norm


K_PARAM = 1

def get_optimal_sigma(k):
    """
        Returns the analytical optimal sigma for a given k.
    """
    return np.sqrt(2*k + 1)

def f(x):
    """
    The pdf of the standard normal distribution
    1/sqrt{2*pi} * exp(-x^2/2)
    """
    return 1/np.sqrt(2*math.pi) * np.exp(-x**2/2)

def g(x, sigma):
    """
    The pdf of the normal distribution with standard deviation sigma
    1/sqrt{2*pi*sigma^2} * exp(-x^2/(2*sigma^2))
    """
    return 1/np.sqrt(2*math.pi*sigma**2) * np.exp(-x**2/(2*sigma**2))

def score_func(X, sigma):
    """
    The score function is partial_{sigma}log f(x,sigma)
    """
    return (1/sigma) * (X**2/sigma**2 - 1)

def likelihood_ratio(X, sigma):
    """
    The likelihood ratio is f(X)/g(X, sigma)
    """
    exponent = -0.5 * X**2 * (1 - 1/sigma**2)
    return sigma * np.exp(exponent)

# Method 1: The "X" Method (LRM)
def m_func(X, sigma, k=K_PARAM):
    """
    m(X, sigma) = f^2(X)/g^2(X,sigma)*X^{4*n}
    = sigma^2 * e^{-x^2*(1-1/sigma^2)}*X^{4*n}
        x = N(0, sigma)
        sigma = the std for g, and we are trying various values to find the optimal
        one that minimizes the variance after importance sampling
        k = a variable that determines the moment: n = 2k, <X^n> = <X^{2k}> is an even moment
    """
    exponent = -np.power(X,2) * (1-1/np.power(sigma,2))
    term1 = np.power(sigma,2) * np.exp(np.clip(exponent, -np.inf, 700))
    term2 = np.power(X, 4*k)
    return term1 * term2

def grad_m_func(x, sigma, m_val, k=K_PARAM):
    """
    grad_m(X, sigma) = L^2(X, sigma)(-g'/g)*X^{4*n}
    = (sigma^2 - X^2/sigma^3) * m(X, sigma)
        x = N(0, sigma)
        sigma = the std for g, and we are trying various values to find the optimal
        one that minimizes the variance after importance sampling
        m_func = the m function and we are finding the gradient as it multiplying some value
        k = a variable that determines the moment: n = 2k, <X^n> = <X^{2k}> is an even moment
    """
    term1 = (np.power(sigma, 2) - np.power(x, 2)) / np.power(sigma, 3)
    return term1 * m_val



# Method 2: The "Z" Method (Reparametrization)
def h_func(Z, sigma, k=K_PARAM):
    """
    h(Z, sigma) = sigma^{4*n+2}*Z^{4*n}*e^{Z^2*(1-sigma^2)}
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
    grad_h(Z, sigma) = ((4*n+2)/sigma - 2*Z^2*sigma)
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

# Batman importance sampling

def f_jax(x):
    """
    The pdf of the standard normal distribution
    1/sqrt{2*pi} * exp(-x^2/2)
    """
    return 1/jnp.sqrt(2*math.pi) * jnp.exp(-x**2/2)

def softmax(t):
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
    return 2*a*z - b*(softmax((z-c)/d) - softmax((-z-c)/d))


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

def Robbins_Monro(z_batch, theta, step_size, f, phi, V, L, score_fn, clip_threshold):
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
        clip_threshold: threshold that controls the upper bound of the value
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
    avg_grad = np.clip(avg_grad, -clip_threshold, clip_threshold)

    # 5. Update theta
    theta_new = theta - step_size * avg_grad

    # 6. CONSTRAINT CHOICES
    a_new, b_new, c_new, d_new = theta_new

    # Underconstruction because the choices of constraints here affect
    # how optimal theta is

    # Constraint A: c is large enough so X doesn't just cluster near 0
    c_new = jnp.maximum(c_new, 0.7)

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