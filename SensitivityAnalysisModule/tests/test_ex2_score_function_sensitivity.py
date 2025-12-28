#Code author: Yunru Zheng
#Date: December 24, 2025

import numpy as np
import math
from src.utils import mean_estimator, mean_sigma
from src.model import score_func


N_SAMPLES = 50000000
SIGMA = 1
RNG = np.random.default_rng(17)

u_theory = lambda sigma: 3*sigma**4
grad_u_theory = lambda sigma: 12*sigma**3

dsigma = [0.2, 0.1, 0.05, 0.02, 0.01]
u_comp = mean_estimator(N_SAMPLES, lambda x: x**4, RNG)
grad_u_comp = mean_estimator(N_SAMPLES, lambda x: x**4*score_func(x,1), RNG)

print(f"baseline u0 = u(sigma0) = {u_comp:.4e}, computed sentivitity is {grad_u_comp:10.4e}")
print(f"   {'dsigma'} | {'u(sigma0+dsigma) (theory)'} | {'u(sigma0+dsigma) (computed)'} | {'du/dsigma'}" )

for i in dsigma:
    new_u_theory = u_theory(SIGMA + i)
    new_u_comp = mean_sigma(N_SAMPLES, lambda x: x**4, RNG, SIGMA + i)

    du_dsigma = (new_u_comp - u_comp) / i

    print(f"{i:9.1e} | {new_u_theory:25.4e} | {new_u_comp:27.4e} | {du_dsigma:.4e}" )





