#Code author: Yunru Zheng
#Date: October 31, 2025

import numpy as np
import math

def var_estimator(N, f, rng):
    """docstring
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
    """docstring
       N = number of
       f = function
       rng = random number
       sigma = chosen std for g
       output = estimated variance for the original sample
    """
    X_g   = sigma * rng.standard_normal(N)
    L_g   = sigma * np.exp(-(X_g**2/2)*(1 - 1/sigma**2))
    Y_g   = f(X_g) * L_g        # new variable after importance sampling
    var_g = np.var(Y_g)
    return var_g
        
#-------  Main program -----------------------

N     = 5000000
k_max = 15       # n = 2k, <X^n> = <X^{2k}> is an even moment

rng   = np.random.default_rng(17)      # instantiate a random number generator object

# Part1: We want to first show that for V(X) = X^10, the change in variance after
# importance Sampling using different sigma values for g.

f         = lambda x, n=10: x**n
sigma_max = 20

#variance before Importance Sampling
var_f1    = var_estimator(N, f, rng)

# tl1 = title line 1
tl1       = f"{'Sigma':>3} | {'Variance':>12} | {'Variance After Importance Sampling':>12}"
print(tl1)
print('-'*100)

for sigma in range(1, sigma_max + 1):
    #new variance after Importance Sampling
    var_g1 = var_new_estimator(N, f, rng, sigma)

    # ol1: output line 1
    ol1 = f"{sigma:6d}| {var_f1:12.2e} | {var_g1:10.2e}"
    print(ol1)

# Part2: Using the optimal sigma value, we want to show the change in variance
# for different moments.

# tl2 = title line 2
tl2 = f"{'n':>4} | {'Variance':>10} | {'Variance After Importance Sampling':>12}"
print(tl2)
print('-'*100)

for k in range(1, k_max + 1):
    n = 2*k
    f2 = lambda x, n=n: x**n

    # Optimal sigma using direct derivation or Laplace Method
    sigma_opt = np.sqrt(n)

    # variance before Importance Sampling
    var_f2 = var_estimator(N, f2, rng)

    # new variance after Importance Sampling
    var_g2 = var_new_estimator(N, f2, rng, sigma_opt)

    # ol2: output line 2
    ol2 = f" {n:3d} | {var_f2:10.2e} | {var_g2:10.2e}"
    print(ol2)




