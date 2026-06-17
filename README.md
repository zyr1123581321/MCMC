Neural Importance Sampling for Monte Carlo Variance Reduction
A framework for reducing the variance of Monte Carlo estimators by learning an optimal importance sampling distribution through a neural change-of-variables, optimized via Robbins–Monro stochastic approximation. Implemented in Python with JAX.
Author: Yunru Zheng  
Advisor: Prof. Jonathan Goodman, Courant Institute of Mathematical Sciences, NYU  
Date: September 2025 – February 2026
---
Overview
Standard Monte Carlo estimation of expectations like E[V(X)] under a distribution f₀ can suffer from high variance, especially when the integrand V(x) is large in regions where f₀ assigns little probability mass. Importance sampling addresses this by drawing samples from a different distribution f(x, θ) and reweighting, but choosing a good f is non-trivial.
This project learns the optimal importance sampling distribution by parameterizing a smooth, invertible transformation φ(z, θ) that maps standard Gaussian samples Z into a new variable X = φ(Z, θ). The parameters θ are optimized to directly minimize the estimator variance using the Robbins–Monro stochastic approximation algorithm, with gradients computed via a score function sensitivity analysis.
Key Contributions
Variance-targeting loss: A custom loss function that directly minimizes Monte Carlo estimator variance rather than a surrogate (e.g., KL divergence).
Score function gradient: Analytical gradient of the variance with respect to the transformation parameters, derived via stochastic sensitivity analysis (no reparameterization trick needed).
Convergence guarantees: Optimization convergence follows from Robbins–Monro stochastic approximation theory.
Stabilization techniques: Mixture sampling with the base distribution, parameter constraints to maintain invertibility, and early stopping with exponential moving average tracking — all developed through systematic experimentation to handle high-dimensional gradient pathologies.
Result: 44% variance reduction on a 20-million-sample benchmark estimating E[Z⁴] under the standard Gaussian.
Method
Neural Change of Variables
The transformation φ : ℝ → ℝ is a skew-symmetric function built from smooth ReLU activations (softplus):
```
φ(z, θ) = φ₊(z) − φ₊(−z)

φ₊(z) = az − b · σ((z − c) / d)

σ(t) = log(1 + eᵗ)       (softplus)
```
The four parameters θ = (a, b, c, d) control:
a — slope near the origin (controls how much mass is moved away from center)
b — magnitude of the slope reduction in the tails
c — location of the transition from steep to shallow slope
d — width of the transition zone
The transformation is guaranteed to be one-to-one when 2a − b/d > 0, which is maintained via parameter constraints during optimization.
Optimization
The loss function is the second moment of the weighted estimator (the variance up to a constant):
```
u(θ) = E_θ[ V(X)² · L(X, θ)² ]
```
where L(x, θ) = f₀(x) / f(x, θ) is the likelihood ratio. Its gradient is computed analytically via the score function:
```
∇_θ u(θ) = −E_θ[ V(X)² · L(X, θ)² · S(X, θ) ]
```
This gradient is estimated from minibatches and used in the Robbins–Monro update:
```
θ_{n+1} = θ_n − t_n · Ĝ_n
```
Mixture Sampling
To prevent the learned distribution from collapsing (assigning near-zero density where f₀ has mass), the sampling distribution is a mixture:
```
h(x, θ) = p · f₀(x) + (1 − p) · f(x, θ)
```
with mixing ratio p (typically 0.1). The score function for the mixture case is derived as:
```
S_mix(x, θ) = β · S₀(x, θ)

β = (1 − p) · g(x, θ) / [p · f₀(x) + (1 − p) · g(x, θ)]
```
Results
The benchmark problem estimates E[Z⁴] = 3 under the standard Gaussian (baseline variance = 96).
Configuration	Optimal θ	Estimator Variance	Variance Reduction
Baseline (no transformation)	—	96.0	—
Learned (with constraints + early stopping)	[1.64, 0.23, 0.54, 0.10]	~13.5	44% relative to the loss objective
Learned (with mixing, p=0.1)	[1.45, 0.19, 0.65, 0.10]	~13.3	Similar, with added robustness
The optimized importance distribution develops a characteristic bimodal ("batman ear") shape, redistributing probability mass from the center toward the tails where V(x) = x⁴ is large — exactly the intuition behind good importance sampling for polynomial moments.
Repository Structure
```
.
├── batman4.py               # Main optimization script
├── src/
│   ├── model.py             # Neural net φ, score functions, likelihood ratios,
│   │                        #   loss function, Robbins–Monro optimizer
│   └── utils.py             # Plotting utilities (distribution, loss curves,
│                            #   gradient histograms, neural net visualization)
├── reports/                  # Weekly progress reports (PDF)
│   ├── Batman_011226.pdf     # Sensitivity analysis module & initial results
│   ├── Batman_011826.pdf     # First batman distribution, sensitivity validation
│   ├── Batman_012326.pdf     # Constraint design: restricting a vs. b
│   ├── Batman_020226.pdf     # Early stopping & first mixing results
│   ├── Batman_020726.pdf     # Mixing derivation correction, p=0.1 vs p=0.5
│   ├── Batman_021726.pdf     # Score function derivation correction, trimming analysis
│   └── Batman_022226.pdf     # Raw gradient histogram analysis, final results
├── Notes.pdf                 # Theoretical notes by Prof. Goodman
└── README.md
```
Usage
Requirements
Python 3.9+
JAX
NumPy, SciPy, Matplotlib
Running
```bash
pip install jax jaxlib numpy scipy matplotlib
python batman4.py
```
The script will:
Initialize the neural net parameters θ = (a, b, c, d)
Run minibatch Robbins–Monro optimization (20M total samples, batch size 10,000)
Track loss with exponential moving average and record the best parameters via early stopping
Plot the learned importance distribution, loss history, neural net transformation, and gradient diagnostics
Key hyperparameters (configured at the top of `batman4.py`):
Parameter	Default	Description
`N_SAMPLES`	20,000,000	Total number of samples
`LEARNING_RATE`	0.0005	Robbins–Monro step size
`MIXING_RATIO`	0.1	Fraction p of base distribution in mixture
`CLIP_THRESHOLD`	None	Gradient clipping (None = disabled)
`TRIMMING_RATIO`	0.0	Fraction of extreme gradients to trim
Mathematical Background
This project extends the framework in Prof. Goodman's notes on Variance Reduction Using Stochastic Approximation (included as `Notes.pdf`), which develops the theory for a scalar twisting parameter. The extension here replaces the scalar with a 4-parameter neural transformation, requiring:
A multivariate score function S(x, θ) with components S_I (change in f₀ due to z shifting) and S_II (change in the Jacobian ϕ_z)
Careful constraint design to maintain invertibility of the transformation
Mixture sampling to handle the expanded parameter space and prevent distributional collapse
Gradient diagnostic tools (histograms at different optimization stages) to understand and address the heavy-tailed gradient distribution
The weekly reports in `reports/` document the full research trajectory, including failed approaches, constraint ablation studies, and the development of stabilization techniques.
License
MIT
Acknowledgments
This project was conducted under the supervision of Prof. Jonathan Goodman at the Courant Institute of Mathematical Sciences, New York University. The theoretical framework and initial model problem formulations are due to Prof. Goodman.
