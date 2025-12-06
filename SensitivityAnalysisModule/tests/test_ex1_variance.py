#Code author: Yunru Zheng
#Date: December 3, 2025

import numpy as np
import math
from src.utils import mean_estimator, mean_new_estimator, var_estimator, var_new_estimator

N_SAMPLES = 50000000
K_MAX = 15       # n = 2k, <X^n> = <X^{2k}> is an even moment

RNG   = np.random.default_rng(17)      # instantiate a random number generator object


# Using the optimal sigma value, we want to show the change in variance
# for different moments.


for k in range(1, K_MAX + 1):
    n = 2*k
    f = lambda x, n=n: x**n

    # Optimal sigma using direct derivation or Laplace Method
    sigma_opt = np.sqrt(n)

    # mean and variance before Importance Sampling
    mean_f = mean_estimator(N_SAMPLES, f, RNG)
    var_f = var_estimator(N_SAMPLES, f, RNG)

    # mean and variance after Importance Sampling
    mean_g = mean_new_estimator(N_SAMPLES, f, RNG, sigma_opt)
    var_g = var_new_estimator(N_SAMPLES, f, RNG, sigma_opt)

    tl = f"For X^{n} using n = {N_SAMPLES:.2e} samples and "
    tl = tl + f"twisted standard deviation r = {sigma_opt:.2e}"
    print(tl)
    print(f"direct estimate is {mean_f:10.4e}, with sample variance {var_f:10.4e}")
    print(f"direct estimate is {mean_g:10.4e}, with sample variance {var_g:10.4e}\n")

# Calculation VERY VERY SLOW!


