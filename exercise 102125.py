import numpy as np
import math

def estimator(N, f):
    X_k = np.random.randn(N)
    Y_k = f(X_k)
    mu_hat = np.mean(Y_k)
    sigma_hat = np.std(Y_k)

    return mu_hat, sigma_hat

def true_mu(n):
    if n % 2 ==1:
        return 0 
    else: 
        return math.prod(range(n-1, 0, -2))

N = 5000000
n_max = 30

print(f"{'n':>3} | {'True Mean':>12} | {'Estimated Mean':>12} | {'Mean Difference':>12} | {'Error Bar':>6}")
print('-'*64)
for n in range(1, n_max + 1):
    f = lambda x, n=n: x**n

    mu = true_mu(n)
    mu_hat, sigma_hat = estimator(N, f)
    mu_difference = abs(mu - mu_hat)
    error_bar = sigma_hat/math.sqrt(N)
    
    print(f"{n:3d} | {mu:12.2e} | {mu_hat:14.2e} | {mu_difference:15.2e} | {error_bar:9.2e}")



