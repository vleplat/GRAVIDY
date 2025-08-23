"""
Objective functions for simplex optimization problems.
Contains least squares and other objectives commonly used with simplex constraints.
"""

import numpy as np


class SimplexLeastSquares:
    """Least squares objective: 0.5 * ||A x - b||^2 on the simplex."""
    
    def __init__(self, A, b):
        """
        Initialize least squares problem.
        
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


def create_test_problem(n, m, cond=5.0, seed=None):
    """Create a test least squares problem on the simplex.
    
    Args:
        n: Dimension of simplex (number of variables)
        m: Number of constraints/observations
        cond: Condition number for the matrix A
        seed: Random seed for reproducibility
        
    Returns:
        A: Matrix (m, n)
        b: Target vector (m,)
        x_star: Ground truth on simplex (n,)
        problem: SimplexLeastSquares instance
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create well-conditioned matrix A
    U, _ = np.linalg.qr(np.random.randn(m, m))
    V, _ = np.linalg.qr(np.random.randn(n, n))
    
    # Singular values with specified condition number
    svals = np.linspace(1.0, cond, min(m, n))
    S = np.zeros((m, n))
    for i, sv in enumerate(svals):
        if i < len(svals):
            S[i, i] = sv
    
    A = U @ S @ V.T
    
    # Ground-truth solution on simplex
    x_star = np.random.dirichlet(np.ones(n))
    
    # Exact target (so x_star is the true solution)
    b = A @ x_star
    
    # Create problem instance
    problem = SimplexLeastSquares(A, b)
    
    return A, b, x_star, problem
