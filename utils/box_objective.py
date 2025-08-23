"""
Objective functions for box-constrained optimization problems.
Contains least squares and other objectives commonly used with box constraints.
"""

import numpy as np


class BoxLeastSquares:
    """Least squares objective: 0.5 * ||A x - b||^2 on box constraints."""
    
    def __init__(self, A, b, lo, hi):
        """
        Initialize box-constrained least squares problem.
        
        Args:
            A: Matrix of shape (m, n)
            b: Target vector of shape (m,)
            lo: Lower bounds of shape (n,)
            hi: Upper bounds of shape (n,)
        """
        self.A = A
        self.b = b
        self.lo = lo
        self.hi = hi
        self.m, self.n = A.shape
        self.H = A.T @ A  # Precompute Hessian (PSD)
        
        # Validate bounds
        if np.any(lo >= hi):
            raise ValueError("Lower bounds must be strictly less than upper bounds")
        
    def f(self, x):
        """Objective value: 0.5 * ||A x - b||^2."""
        r = self.A @ x - self.b
        return 0.5 * float(r @ r)
    
    def grad(self, x):
        """Gradient: A^T(Ax - b)."""
        return self.A.T @ (self.A @ x - self.b)
    
    def hess(self):
        """Hessian: A^T A (constant, PSD)."""
        return self.H
    
    def residual(self, x):
        """Residual vector: Ax - b."""
        return self.A @ x - self.b
    
    def project(self, x):
        """Project x onto the box constraints."""
        return np.minimum(np.maximum(x, self.lo), self.hi)


def create_box_test_problem(n, m, seed=None):
    """Create a test least squares problem with box constraints.
    
    Args:
        n: Dimension of the problem
        m: Number of constraints/observations
        seed: Random seed for reproducibility
        
    Returns:
        A: Matrix (m, n)
        b: Target vector (m,)
        lo: Lower bounds (n,)
        hi: Upper bounds (n,)
        x_star: Ground truth solution
        problem: BoxLeastSquares instance
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create box bounds
    lo = -1.0 + 0.2 * np.random.randn(n)
    hi = 1.0 + 0.2 * np.random.randn(n)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    
    # Create mildly well-conditioned matrix
    A = np.random.randn(m, n) / np.sqrt(n) + 0.1 * np.eye(m, n)
    
    # Ground-truth solution within box
    x_star = lo + (hi - lo) * np.random.rand(n)
    
    # Exact target (so x_star is the true solution)
    b = A @ x_star
    
    # Create problem instance
    problem = BoxLeastSquares(A, b, lo, hi)
    
    return A, b, lo, hi, x_star, problem
