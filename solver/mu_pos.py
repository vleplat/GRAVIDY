"""
Multiplicative Updates (MU) solver for positive orthant optimization.
Baseline method that requires A>=0, b>=0 and keeps positivity automatically.
"""

import numpy as np
import time


def MU_pos(problem, max_iters=400, x0=None, eps=1e-16, tol_grad=1e-8, verbose=False):
    """
    Multiplicative Updates (MU) solver for positive orthant optimization.
    
    MU baseline: x <- x * (A^T b) / (A^T A x + eps)
    Requires A>=0, b>=0. Keeps positivity automatically.
    
    Args:
        problem: PositiveLeastSquares problem instance (with A>=0, b>=0)
        max_iters: Maximum iterations
        x0: Initial point (if None, uses random positive start)
        eps: Small constant to avoid division by zero
        tol_grad: Convergence tolerance
        verbose: Print iteration progress
        
    Returns:
        x: Final solution
        history: List of (iter, obj_val, grad_norm, time) tuples
    """
    A = problem.A
    b = problem.b
    n = problem.n
    
    # Check if MU is applicable (A>=0, b>=0)
    if np.any(A < 0) or np.any(b < 0):
        raise ValueError("Multiplicative Updates require A>=0 and b>=0")
    
    # Initialize with random positive start
    if x0 is None:
        x = np.maximum(np.random.rand(n), 1e-6)
    else:
        x = np.maximum(x0, 1e-12)

    Atb = A.T @ b
    H = A.T @ A
    
    history = []
    t0 = time.time()
    
    for k in range(max_iters):
        # Compute objective and gradient
        obj_val = problem.f(x)
        grad = problem.grad(x)
        grad_norm = np.linalg.norm(grad)
        current_time = time.time() - t0
        
        history.append((k, obj_val, grad_norm, current_time))
        
        if verbose and k % 50 == 0:
            print(f"[MU-pos] iter={k:4d} f={obj_val:.6e} ||grad||={grad_norm:.3e} time={current_time:.2f}s")
        
        # Check convergence
        if grad_norm <= tol_grad:
            break
        
        # Multiplicative update
        Hx = H @ x + eps
        x = x * (Atb / Hx)
    
    return x, history
