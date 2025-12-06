#Code author: Yunru Zheng
#Date: December 6, 2025

import numpy as np
from src.utils import var_estimator, var_new_estimator
from src.model import f, g, m_func, grad_m_func

N_SAMPLES = 50000000
SIGMA = 1
RNG = np.random.default_rng(17)

#u_var_comp = var_estimator(N_SAMPLES, lambda x: x**4, RNG)

print(g(0,2))
print(g(1,2))
