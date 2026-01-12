#Code author: Yunru Zheng
#Date: January 6, 2025

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import grad


N_SAMPLES = 200000
RNG = np.random.default_rng(17)

def h(t):
    return np.log(1 + np.exp(t))

def phi(z, a, b, c, d):
    return 2*a*z - b*(h((z-c)/d) - h((-z-c)/d))


a = 1.0
b = 0.45
c = 2.0
d = 0.3

Z = RNG.standard_normal(N_SAMPLES)
X = phi(Z, a, b, c, d)


Z_space = np.linspace(-15.0, 15.0, 100)


plt.figure(figsize=(10, 6))
plt.plot(Z_space, phi(Z_space, a, b, c, d), label='estimation variance')

plt.title(f"The effect of the neural net on Z (N={N_SAMPLES})")
plt.xlabel("Z")
plt.ylabel("X")
plt.grid(True, alpha=0.3)

plt.legend()
plt.show()


plt.figure(figsize=(10, 6))
plt.hist(X, bins=100, density=True, alpha=0.5, label="neural net")
plt.title(f"The distribution of the neural net (N={N_SAMPLES})")
plt.xlabel("X")
plt.ylabel("Density")
plt.grid(True, alpha=0.3)

plt.legend()
plt.show()
