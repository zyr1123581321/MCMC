
import numpy as np
import math
import jax
import jax.numpy as jnp
from jax import grad
from src.utils import mean_estimator, mean_sigma


N_SAMPLES = 50000000
SIGMA = 1
RNG = np.random.default_rng(17)


def S(x,sigma):
    return (1/sigma) * (x**2/sigma**2 - 1)

u_theory = lambda sigma: 3*sigma**4
grad_u_theory = grad(u_theory)


dsigma = [0.2, 0.1, 0.05, 0.02, 0.01]
u_comp = mean_estimator(N_SAMPLES, lambda x: x**4, RNG)
grad_u_comp = mean_estimator(N_SAMPLES, lambda x: x**4*S(x,1), RNG)

print(f"baseline u0 = u(sigma0) = {u_comp:.4e}, computed sentivitity is {grad_u_comp:10.4e}")
print(f"   {'dr':>4} | {'u(sigma0+dsigma) (theory)'} | {'u(sigma0+dsigma) (computed)'} | {'du/dr'}" )

for i in dsigma:
    new_u_theory = u_theory(SIGMA + i)
    new_u_comp = mean_sigma(N_SAMPLES, lambda x: x**4, RNG, SIGMA + i)

    dudsigma = (new_u_comp - u_comp) / i

    print(f"{i:6.1e} | {new_u_theory:25.4e} | {new_u_comp:27.4e} | {dudsigma:.4e}" )





