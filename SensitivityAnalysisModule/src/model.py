#Code author: Yunru Zheng
#Date: December 3, 2025

import numpy as np
import math

K_PARAM = 1

def get_optimal_sigma(k):
    """
        Returns the analytical optimal sigma for a given k.
    """
    return np.sqrt(2*k + 1)

def f(x):
    """
    The pdf of the standard normal distribution
    1/sqrt{2*pi} * exp(-x^2/2)
    """
    return 1/np.sqrt(2*math.pi) * np.exp(-x**2/2)

def g(x, sigma):
    """
    The pdf of the normal distribution with standard deviation sigma
    1/sqrt{2*pi*sigma^2} * exp(-x^2/(2*sigma^2))
    """
    return 1/np.sqrt(2*math.pi*sigma**2) * np.exp(-x**2/(2*sigma**2))

def score_func(X, sigma):
    """
    The score function is partial_{sigma}log f(x,sigma)
    """
    return (1/sigma) * (X**2/sigma**2 - 1)

def likelihood_ratio(X, sigma):
    """
    The likelihood ratio is f(X)/g(X, sigma)
    """
    exponent = -0.5 * X**2 * (1 - 1/sigma**2)
    return sigma * np.exp(exponent)

# Method 1: The "X" Method (LRM)
def m_func(X, sigma, k=K_PARAM):
    """
    m(X, sigma) = f^2(X)/g^2(X,sigma)*X^{4*n}
    = sigma^2 * e^{-x^2*(1-1/sigma^2)}*X^{4*n}
        x = N(0, sigma)
        sigma = the std for g, and we are trying various values to find the optimal
        one that minimizes the variance after importance sampling
        k = a variable that determines the moment: n = 2k, <X^n> = <X^{2k}> is an even moment
    """
    exponent = -np.power(X,2) * (1-1/np.power(sigma,2))
    term1 = np.power(sigma,2) * np.exp(np.clip(exponent, -np.inf, 700))
    term2 = np.power(X, 4*k)
    return term1 * term2

def grad_m_func(x, sigma, m_val, k=K_PARAM):
    """
    grad_m(X, sigma) = L^2(X, sigma)(-g'/g)*X^{4*n}
    = (sigma^2 - X^2/sigma^3) * m(X, sigma)
        x = N(0, sigma)
        sigma = the std for g, and we are trying various values to find the optimal
        one that minimizes the variance after importance sampling
        m_func = the m function and we are finding the gradient as it multiplying some value
        k = a variable that determines the moment: n = 2k, <X^n> = <X^{2k}> is an even moment
    """
    term1 = (np.power(sigma, 2) - np.power(x, 2)) / np.power(sigma, 3)
    return term1 * m_val



# Method 2: The "Z" Method (Reparametrization)
def h_func(Z, sigma, k=K_PARAM):
    """
    h(Z, sigma) = sigma^{4*n+2}*Z^{4*n}*e^{Z^2*(1-sigma^2)}
        Z = standard normal distributed variable
        sigma = the std for g, and we are trying various values to find the optimal
        one that minimizes the variance after importance sampling
        k = a variable that determines the moment: n = 2k, <X^n> = <X^{2k}> is an even moment
    """
    term1 = np.power(sigma, 4*k + 2)
    term2 = np.power(Z, 4*k)
    exponent = np.power(Z, 2) * (1 - np.power(sigma, 2))
    term3 = np.exp(np.clip(exponent, -np.inf, 700))

    return term1 * term2 * term3

def grad_h_func(Z, sigma, h_func, k=K_PARAM):
    """
    grad_h(Z, sigma) = ((4*n+2)/sigma - 2*Z^2*sigma)
        Z = standard normal distributed variable
        sigma = the std for g, and we are trying various values to find the optimal
        one that minimizes the variance after importance sampling
        h_func = the h function and we are finding the gradient as it multiplying some value
        k = a variable that determines the moment: n = 2k, <X^n> = <X^{2k}> is an even moment
    """

    term1 = (4*k + 2) / sigma
    term2 = 2 * sigma * np.power(Z, 2)
    gradient = (term1 - term2) * h_func
    return gradient

