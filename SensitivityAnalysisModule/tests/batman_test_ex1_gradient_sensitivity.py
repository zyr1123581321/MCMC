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


def sensitivity_check(theta, n_samples=50000000):
    """
    We are comparing the computed gradient loss with the secant value calculated from the variances to make sure that the functions in model.py are correct
    Input:
        theta: Parameters for the transformation function
        n_samples: number of samples used
    """

    a, b, c, d = theta
    # Setup functions for later use
    fast_score_fn = make_score_function(phi)
    fast_L_fn = make_likelihood_ratio(f_jax, phi)

    z_batch = RNG.standard_normal(n_samples)

    # The function we test on is X^4
    V = lambda z: z**4

    # Finding the baseline variance for X^4
    baseline_loss = jnp.mean(loss_function(z_batch, theta, f_jax, phi, V, fast_L_fn))
    u_var_comp = baseline_loss - 9.0

    # finding the average gradient loss using defined function
    grad_loss = gradient_loss(z_batch, theta, f_jax, phi, V, fast_L_fn, fast_score_fn)
    comp_sensitivity = jnp.mean(grad_loss, axis=0)

    print("Computing Analytical Gradient (via gradient_loss function)")
    print(f"Starting from (theta = [{a:.4f}, {b:.4f}, {c:.4f}, {d:.4f}])")
    print(f"baseline variance = {u_var_comp:.4e}, baseline loss = {float(baseline_loss):.4e}")

    # Printing jax arrays as np arrays
    grad_str = [f"{x:.4e}" for x in np.array(comp_sensitivity)]
    print(f"computed sentivitity is {grad_str}")



    param_names = ['a', 'b', 'c', 'd']
    increments = [0.1, 0.01, 0.005, 0.002, 0.001]

    for incr in increments:

        print(f"increments {incr:.4e}")
        print(f"{'parameters':>4} | {'ssq(theta0+dtheta) (computed)'} | {'dssq/dtheta'}")
        for i, name in enumerate(param_names):
            val_base = theta[i]

            # Calculate epsilon value with respect to each individual parameter
            if abs(val_base) < 1e-9:
                eps = incr
            else:
                eps = val_base * incr

            # Generate new theta value
            theta_new = theta + jnp.zeros(4).at[i].set(eps)

            # calculate new ssq (loss - expectation * 2)
            loss_val = loss_function(z_batch, theta_new, f_jax, phi, V, fast_L_fn)
            ssq = np.mean(loss_val) - 9.0
            dssq_dtheta = (ssq - u_var_comp) / eps

            print(f"{name:>10} | {float(ssq):29.4e} | {float(dssq_dtheta):.4e}" )


#---------------------------Main------------------------------


#a, b, c, d = 0.5, 0, 2.0, 1.0
#a, b, c, d = 1.5955062, 0.45257553, 2.5, 0.20656002
theta1 = jnp.array([0.5, 0, 2.0, 1.0])
sensitivity_check(theta1)

theta2 = jnp.array([2.0, 0.45, 2.0, 0.15])

sensitivity_check(theta2)

