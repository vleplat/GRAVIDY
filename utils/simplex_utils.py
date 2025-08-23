"""
Simplex utilities for GRAVIDY–Δ (simplex) solver.
Contains softmax, Jacobian, projection, and other simplex-related operations.
"""

import numpy as np


def softmax(u):
    """Numerically stable softmax on R^n (returns a probability vector)."""
    um = np.max(u)
    e = np.exp(u - um)
    s = e / np.sum(e)
    return s


def jac_softmax(u):
    """Jacobian of softmax at u: diag(s) - s s^T."""
    s = softmax(u)
    return np.diag(s) - np.outer(s, s)


def project_simplex(v):
    """Euclidean projection onto the simplex {x >= 0, sum x = 1}.
    
    Uses the algorithm from Duchi et al. (2008).
    
    Args:
        v: Vector to project onto simplex
        
    Returns:
        Projected vector on the probability simplex
    """
    n = len(v)
    u = np.sort(v)[::-1]  # Sort in descending order
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    return w


def center_logits(u):
    """Center logits by subtracting mean (removes gauge freedom)."""
    return u - np.mean(u)


def uniform_simplex(n):
    """Return uniform distribution on n-simplex."""
    return np.ones(n) / n


def random_simplex(n, seed=None):
    """Generate random point on simplex using Dirichlet distribution."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.dirichlet(np.ones(n))


def safe_log(x, eps=1e-16):
    """Safe logarithm that clips values near zero."""
    return np.log(np.clip(x, eps, 1.0))


def kl_divergence(x, p):
    """KL divergence KL(x || p) between two probability vectors."""
    x_safe = np.clip(x, 1e-16, 1.0)
    p_safe = np.clip(p, 1e-16, 1.0)
    return float(np.sum(x_safe * (np.log(x_safe) - np.log(p_safe))))
