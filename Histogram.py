#Code author: Yunru Zheng
#Date: November 28, 2025

import numpy as np
import matplotlib.pyplot as plt
import math

RNG = np.random.default_rng(17)
N_SAMPLES = 50000

def histogram(sample, sample_name="Distribution", num_bins=30):
    """
        sample: data, assuming in the form of a numpy array
        sample_name: the specific distribution
        n_bins: the number of total bins
    """

    # 1. Finding the range of the sample and the bin width based on
    # the range and the number of bins
    min_val = np.min(sample)
    max_val = np.max(sample)
    data_range = max_val - min_val

    # 2. Finding the bin width
    # Formula: delta = (b - a) / n
    bin_width = data_range / num_bins

    # 3. Initiate the bins
    counts = [0] * num_bins

    # 4. Putting the sample data into bins
    for data in sample:
        # calculating the bin index that data belongs to
        # Formula: index = (sample - a) / delta
        if data == max_val:
            index = num_bins - 1
        else:
            index = math.floor((data - min_val) / bin_width)

        counts[index] += 1

    # 5. Finding the bin centers
    bin_center = [0] * num_bins
    # x_{k+1/2} = a + delta * (k + 1/2)
    for i in range(num_bins):
        bin_center [i] = min_val + bin_width * (i+1/2)

    density = [c / (bin_width * len(sample)) for c in counts]

    # 6. Plotting
    plt.figure()

    plt.bar(bin_center, density, bin_width)
    plt.plot(bin_center, density, 'o', color='black', markersize=6, zorder=10,
        linewidth=1, label='Density')
    plt.title(f'Histogram of {sample_name}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

def plot_with_pdf(sample, pdf_func, sample_name, num_bins=30):
    """
    Since the histogram function doesn't have access to the actual pdf, I need
    a helper function to plot the pdf on top of it
        sample: data, assuming in the form of a numpy array
        pdf_func: The actual pdf function
        sample_name: the specific distribution
        n_bins: the number of total bins
    """
    # 1. Call the histogram function
    histogram(sample, sample_name, num_bins)

    # 2. Generate x axis
    x_min, x_max = np.min(sample), np.max(sample)
    x_fine = np.linspace(x_min, x_max, 1000)

    # 3. Generate y values
    y_fine = pdf_func(x_fine)

    plt.plot(x_fine, y_fine, 'r-', linewidth=2, label='Theoretical PDF')
    plt.legend()

    plt.show()


def gaussian_from_exponential(n_samples):
    """
        n_samples: the total number of samples taken
    """
    samples = []
    while len(samples) < n_samples:
        # 1. Generate 1/2 * e^{-|x|}
        E = RNG.exponential()
        S = RNG.choice([-1, 1])
        X = S * E

        # Generate the uniform distribution
        U = RNG.uniform()

        # 2. Rejection
        # By hand calculation, B_h = max f(x)/g(x) = \sqrt{2*e/pi}
        # The acceptance rate is exp(-1/2*(|X|-1)^2)
        # we take the logarithm of both sizes to simplify the inequality
        if np.log(U) < -0.5 * np.power(abs(X)-1,2):
            samples.append(X)
    return np.array(samples)


#-------  Main program -----------------------

# Sample 1: Exponential distribution
s2 = RNG.exponential(size=N_SAMPLES)
plot_with_pdf(s2, lambda x: np.exp(-x), "Exponential Distribution")


# Sample 2: Standard Gaussian distribution
s1 = RNG.standard_normal(N_SAMPLES)
def pdf_gaussian(x):
    return (1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2))

plot_with_pdf(s1, pdf_gaussian, "Standard Normal Distribution")



# Playing around for each function
# Commented out eventually for better performance of the code
"""
s3 = RNG.gamma(5, 0.5, size=N_SAMPLES)
histogram(s3, "Gamma Distribution")

s4 = RNG.uniform(2,3, size=N_SAMPLES)
histogram(s4, "Uniform Distribution")
"""

# The following two should generate the same histogram with the distribution
# 1/2 * e^{-|x|}
"""
s5 = np.random.choice([-1, 1], size=N_SAMPLES) * RNG.exponential(size=N_SAMPLES)
histogram(s5, "Laplace Distribution1")

s6 = RNG.exponential(size=N_SAMPLES) - RNG.exponential(size=N_SAMPLES)
histogram(s6, "Laplace Distribution2")
"""

# Sample 3: Standard Gaussian from exponential distribution using rejection methods
s_gaussian = gaussian_from_exponential(N_SAMPLES)
plot_with_pdf(s_gaussian, pdf_gaussian, "Gaussian Distribution from Exponential Distribution Using Rejection Methods")

