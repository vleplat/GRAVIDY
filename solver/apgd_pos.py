"""
Accelerated Projected Gradient Descent (APGD) with Nesterov acceleration for positive orthant.
External baseline method for comparison with GRAVIDY–pos.
"""

import numpy as np
import time


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


def APGD_pos(problem, max_iters=400, step_size=None, x0=None, tol_grad=1e-8, verbose=False):
    """
    Accelerated Projected Gradient Descent with Nesterov acceleration for positive orthant.
    
    Uses proper Nesterov acceleration with Lipschitz constant estimation.
    
    Args:
        problem: PositiveLeastSquares problem instance
        max_iters: Maximum iterations
        step_size: Step size (if None, auto-estimate Lipschitz constant)
        x0: Initial point (if None, uses random nonnegative start)
        tol_grad: Convergence tolerance
        verbose: Print iteration progress
        
    Returns:
        x: Final solution
        history: List of (iter, obj_val, grad_norm, time) tuples
    """
    n = problem.n
    
    # Initialize with random nonnegative start
    if x0 is None:
        x = np.maximum(np.random.randn(n), 0.0)
    else:
        x = np.maximum(x0, 0.0)
    
    # Estimate Lipschitz constant if not provided
    if step_size is None:
        # For least squares: L = ||A^T A||_2
        A = problem.A
        H = lambda v: A.T @ (A @ v)
        L = power_iteration_lmax(H, n, iters=60)
        step_size = 1.0 / L
        if verbose:
            print(f"[APGD-pos] Estimated Lipschitz constant: {L:.6f}, step size: {step_size:.6f}")
    
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
            print(f"[APGD-pos] iter={k:4d} f={obj_val:.6e} ||grad||={grad_norm:.3e} time={current_time:.2f}s")
        
        # Check convergence
        if grad_norm <= tol_grad:
            break
        
        # Gradient step with projection (clipping to nonnegative)
        x_old = x.copy()
        x = np.maximum(y - step_size * grad, 0.0)
        
        # Update momentum parameter (Nesterov acceleration)
        t_old = t
        t = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_old * t_old))
        
        # Update momentum point
        y = x + ((t_old - 1.0) / t) * (x - x_old)
    
    return x, history
