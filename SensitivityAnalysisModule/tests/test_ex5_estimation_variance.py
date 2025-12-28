import numpy as np
import matplotlib.pyplot as plt
from src.utils import var_new_estimator



N_SAMPLES = 50000
RNG = np.random.default_rng(17)


sigma_values = np.linspace(1.0, 10.0, 100)
variances = []

f = lambda x: x**4

for sigma in sigma_values:
    v = var_new_estimator(N_SAMPLES, f, RNG, sigma)
    variances.append(v)

min_variance = np.min(variances)
min_index = np.argmin(variances)
sigma_optimal = sigma_values[min_index]

plt.figure(figsize=(10, 6))
plt.plot(sigma_values, variances, label='estimation variance')

plt.title("Scalar Gaussian example, estimation variance")
plt.xlabel("twisting parameter, sigma")
plt.ylabel("variance")
plt.axhline(y=min_variance, linestyle='--', label=f'Min Variance ({min_variance:.2f})')
plt.axvline(x=sigma_optimal, linestyle=':', label=f'Optimal sigma ({sigma_optimal:.2f})')
plt.grid(True, alpha=0.3)


plt.legend()
plt.show()