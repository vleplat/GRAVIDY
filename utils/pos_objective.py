"""
Objective functions for positive orthant optimization problems (NNLS).
Contains least squares and other objectives commonly used with non-negativity constraints.
"""

import numpy as np


class PositiveLeastSquares:
    """Least squares objective: 0.5 * ||A x - b||^2 on positive orthant (x >= 0)."""
    
    def __init__(self, A, b):
        """
        Initialize positive orthant least squares problem.
        
        Args:
            A: Matrix of shape (m, n)
            b: Target vector of shape (m,)
        """
        self.A = A
        self.b = b
        self.m, self.n = A.shape
        self.H = A.T @ A  # Precompute Hessian (PSD)
        
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
        """Project x onto the positive orthant."""
        return np.maximum(x, 0.0)


def create_pos_test_problem(n, m, density=0.2, seed=None):
    """Create a test least squares problem with positive orthant constraints.
    
    Args:
        n: Dimension of the problem
        m: Number of constraints/observations
        density: Sparsity density of ground truth solution (0 < density <= 1)
        seed: Random seed for reproducibility
        
    Returns:
        A: Matrix (m, n) with A >= 0
        b: Target vector (m,) with b >= 0
        x_star: Ground truth solution (sparse, nonnegative)
        problem: PositiveLeastSquares instance
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create nonnegative matrix A
    A = np.abs(np.random.randn(m, n)) + 0.05 * np.eye(m, n)
    
    # Create sparse nonnegative ground truth
    x_star = np.zeros(n)
    k = max(1, int(density * n))
    supp = np.random.choice(n, k, replace=False)
    x_star[supp] = np.abs(np.random.randn(k)) + 0.1
    
    # Exact target (so x_star is the true solution)
    b = A @ x_star
    
    # Ensure b >= 0 (add small positive constant if needed)
    if np.any(b < 0):
        b = b - np.min(b) + 0.1
    
    # Create problem instance
    problem = PositiveLeastSquares(A, b)
    
    return A, b, x_star, problem
