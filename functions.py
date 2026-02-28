import numpy as np
from scipy.stats import truncnorm
from config import a, b, c, d


def f_poly(x):
    return a * x**3 + b * x**2 + c * x + d


def f_sin(x):
    return x * np.sin(2 * np.pi * x)


def generate_error_uniform(n, eps0):
    return np.random.uniform(-eps0, eps0, size=n)


def generate_error_normal(n, eps0):
    sigma = eps0 / 3
    a_trunc = -eps0 / sigma
    b_trunc = eps0 / sigma
    return truncnorm.rvs(a_trunc, b_trunc, scale=sigma, size=n)


def generate_sample(f, n, eps0, error_type='uniform'):
    x = np.random.uniform(-1, 1, size=n)
    
    if error_type == 'uniform':
        eps = generate_error_uniform(n, eps0)
    else:
        eps = generate_error_normal(n, eps0)
    
    y = f(x) + eps
    return x, y
