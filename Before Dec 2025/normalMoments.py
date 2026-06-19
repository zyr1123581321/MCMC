

import numpy as np
import math

def estimator(N, f, rng):
    """docstring
       N = number of 
       f = function
       rng = random number ..
       output = 
    """
    X = rng.standard_normal(N)   # N independent standard normals
    Y = f(X)
    mu_hat = np.mean(Y)
    sigma_hat = np.std(Y)

    return mu_hat, sigma_hat

def true_mu(n):
    if n % 2 ==1:
        return 0 
    else: 
        return math.prod(range(n-1, 0, -2))
        
#-------  Main program -----------------------

N     = 5000000
k_max = 15       # n = 2k, <X^n> = <X^{2k}> is an even moment

#  see https://numpy.org/doc/stable/reference/random/index.html

rng = np.random.default_rng(17)      # instantiate a random number generator object

tl =      "  n |    True Mean | Estimated Mean | Mean Difference |"   # tl = title line
tl = tl + " Relative Difference |  Error Bar |  n_sigma "
print(tl)
print('-'*100)
for k in range(1, k_max + 1):
    n = 2*k
    f = lambda x, n=n: x**n

    mu = true_mu(n)
    mu_hat, sigma_hat = estimator(N, f, rng)
    mu_difference = mu - mu_hat
    rel_difference = mu_difference/mu
    error_bar = sigma_hat/math.sqrt(N)
    n_sig     = mu_difference/error_bar
    
    ol =      f"{n:3d} | {mu:10.2e}   |  {mu_hat:10.2e}    |"         # output line
    ol = ol + f"  {mu_difference:12.2e}   |    {rel_difference:10.2e}       |"
    ol = ol + f" {error_bar:9.2e}  |{n_sig:7.2f}"
    print(ol)



