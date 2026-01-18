#Code author: Yunru Zheng
#Date: December 6, 2025

import numpy as np
from src.utils import var_estimator, var_new_estimator
from src.model import m_func, grad_m_func

N_SAMPLES = 50000000
SIGMA = 1
RNG = np.random.default_rng(17)

# The function we test on is X^4
# Finding the baseline variance for X^4
u_var_comp = var_estimator(N_SAMPLES, lambda x: x**4, RNG)

X_base = RNG.standard_normal(N_SAMPLES)

m_val = m_func(X_base, SIGMA, 2)

# average gradient loss using defined function
comp_sensitivity = np.mean(grad_m_func(X_base, SIGMA, m_val, 2))

print(f"baseline variance = {u_var_comp:.4e}, computed sentivitity is {comp_sensitivity:10.4e}")
print(f"   {'dsigma':>4} | {'ssq(sigma0+dsigma) (computed)'} | {'dssq/dsigma'}" )


# Setting increments for sigma
dsigma = [0.1, 0.01, 0.005, 0.002, 0.001]

for i in dsigma:
    sigma_new = SIGMA + i
    X_new = RNG.standard_normal(N_SAMPLES) * sigma_new

    # ssq = mean loss - expectation^2 (9 here for X^4)
    ssq = np.mean(m_func(X_new, sigma_new, 2)) - 9.0

    # finding the dssq/dsigma by finding the tangent
    dssq_dsigma = (ssq - u_var_comp) / i
    print(f"{i:9.1e} | {ssq:29.4e} | {dssq_dsigma:.4e}" )

