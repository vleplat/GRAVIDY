"""
Accelerated Projected Gradient Descent (APGD) with Nesterov acceleration on the simplex.
External baseline method for comparison with GRAVIDY–Δ.
"""

import numpy as np
import time
from utils.simplex_utils import project_simplex


def power_iteration_lmax(Mv, n, iters=50):
    """Estimate largest eigenvalue of a PSD matrix via matrix-vector product Mv(v)"""
    v = np.random.randn(n)
    v /= np.linalg.norm(v) + 1e-16
    lam = None
    for _ in range(iters):
        w = Mv(v)
        lam_new = float(v @ w)
        wn = np.linalg.norm(w)
        if wn == 0:
            break
        v = w / wn
        lam = lam_new
    if lam is None or not np.isfinite(lam) or lam <= 0:
        lam = 1.0
    return lam


def APGD_simplex(problem, max_iters=400, step_size=None, x0=None, tol_grad=1e-8, verbose=False):
    """
    Accelerated Projected Gradient Descent with Nesterov acceleration on the simplex.
    
    Uses proper Nesterov acceleration with Lipschitz constant estimation.
    
    Args:
        problem: Objective with f(), grad() methods
        max_iters: Maximum iterations
        step_size: Step size (if None, auto-estimate Lipschitz constant)
        x0: Initial point on simplex (if None, uses uniform)
        tol_grad: Convergence tolerance
        verbose: Print iteration progress
        
    Returns:
        x: Final solution on simplex
        history: List of (iter, obj_val, grad_norm, time) tuples
    """
    n = problem.n
    
    # Initialize on simplex
    if x0 is None:
        x = np.ones(n) / n  # Uniform distribution
    else:
        x = project_simplex(x0)
    
    # Estimate Lipschitz constant if not provided
    if step_size is None:
        # For least squares: L = ||A^T A||_2
        A = problem.A  # Access the matrix directly
        H = lambda v: A.T @ (A @ v)
        L = power_iteration_lmax(H, n, iters=60)
        step_size = 1.0 / L
        if verbose:
            print(f"[APGD] Estimated Lipschitz constant: {L:.6f}, step size: {step_size:.6f}")
    
    # Initialize momentum variables
    y = x.copy()  # Momentum point
    t = 1.0       # Momentum parameter
    
    history = []
    t0 = time.time()
    
    for k in range(max_iters):
        # Compute objective and gradient at momentum point
        obj_val = problem.f(y)
        grad = problem.grad(y)
        grad_norm = np.linalg.norm(grad)
        current_time = time.time() - t0
        
        history.append((k, obj_val, grad_norm, current_time))
        
        if verbose and k % 50 == 0:
            print(f"[APGD] iter={k:4d} f={obj_val:.6e} ||grad||={grad_norm:.3e} time={current_time:.2f}s")
        
        # Check convergence
        if grad_norm <= tol_grad:
            break
        
        # Gradient step with projection
        x_old = x.copy()
        x = project_simplex(y - step_size * grad)
        
        # Update momentum parameter (Nesterov acceleration)
        t_old = t
        t = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_old * t_old))
        
        # Update momentum point
        y = x + ((t_old - 1.0) / t) * (x - x_old)
    
    return x, history
