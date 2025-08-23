"""
Entropic Mirror Descent (EMD) on the simplex.
External baseline method for comparison with GRAVIDY–Δ.
"""

import numpy as np
import time


def EMD_simplex(problem, max_iters=400, step_size=1.0, armijo_c=1e-4, 
                backtrack_beta=0.5, x0=None, tol_grad=1e-8, verbose=False):
    """
    Entropic Mirror Descent on the simplex with multiplicative updates.
    
    Uses the entropic mirror map with backtracking line search.
    
    Args:
        problem: Objective with f(), grad() methods
        max_iters: Maximum iterations
        step_size: Initial step size
        armijo_c: Armijo line search parameter
        backtrack_beta: Backtracking factor
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
        x = x0.copy()
    
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
            print(f"[EMD] iter={k:4d} f={obj_val:.6e} ||grad||={grad_norm:.3e} time={current_time:.2f}s")
        
        # Check convergence
        if grad_norm <= tol_grad:
            break
        
        # Armijo backtracking with entropic updates
        t = step_size
        fx = obj_val
        
        while True:
            # Entropic mirror update (multiplicative)
            y = x * np.exp(-t * grad)
            y /= np.sum(y)  # Normalize to simplex
            
            fy = problem.f(y)
            
            # Armijo condition: sufficient decrease
            if fy <= fx - armijo_c * t * np.dot(grad, x - y) or t < 1e-16:
                break
            
            t *= backtrack_beta
        
        x = y
    
    return x, history
